"""GPU telemetry logger and isolated per-variant runner for Experiment 2.

Collects high-frequency user-space GPU telemetry via ``nvidia-smi`` and Linux
sysfs PCIe Advanced Error Reporting (AER) counters without requiring root
privileges. Flushes and ``fsync``s every sample to disk so pre-crash states are
preserved even if the GPU falls off the bus or enters ``ERR!`` state.
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PCI_DEVICE_PATH = Path("/sys/bus/pci/devices/0000:01:00.0")
DEFAULT_NVIDIA_PROC_PATH = Path("/proc/driver/nvidia/gpus/0000:01:00.0")
DEFAULT_OUTPUT_DIR = Path("Outputs/Experiment_2_variants")

NVIDIA_SMI_FIELDS: tuple[str, ...] = (
    "timestamp",
    "pstate",
    "gpu_recovery_action",
    "power.draw.instant",
    "power.draw.average",
    "enforced.power.limit",
    "temperature.gpu",
    "temperature.gpu.tlimit",
    "temperature.memory",
    "fan.speed",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.reserved",
    "clocks.current.graphics",
    "clocks.current.memory",
    "pcie.link.gen.current",
    "pcie.link.width.current",
    "clocks_event_reasons.active",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons_counters.sw_power_cap",
    "clocks_event_reasons_counters.hw_thermal_slowdown",
    "clocks_event_reasons_counters.hw_power_brake_slowdown",
)

ALL_EXP1A_VARIANTS: tuple[str, ...] = tuple(f"exp1a_variant{i}" for i in range(1, 11))
ALL_EXP2_MODELS: tuple[str, ...] = (
    "barlowtwins",
    "resnet152",
    "densenet161",
    "efficientnetb3",
    "vgg16",
    "vgg19",
    "clip_vit_b16",
)


@dataclass(frozen=True)
class TelemetrySample:
    """Single point-in-time snapshot of GPU, PCIe, and experiment state."""

    host_utc_iso: str
    active_variant: str
    completed_mpm_variants: int
    active_temp_parts: int
    gpu_healthy: bool
    anomaly_reason: str
    pcie_link_speed_sysfs: str
    pcie_link_width_sysfs: str
    pcie_power_state_sysfs: str
    aer_correctable_total: int
    aer_nonfatal_total: int
    aer_fatal_total: int
    nvidia_smi_values: dict[str, str]

    def to_csv_row(self) -> dict[str, str | int | bool]:
        """Convert the sample into a flat dictionary matching CSV headers.

        Returns:
            Flat mapping of column names to primitive values.
        """
        row: dict[str, str | int | bool] = {
            "host_utc_iso": self.host_utc_iso,
            "active_variant": self.active_variant,
            "completed_mpm_variants": self.completed_mpm_variants,
            "active_temp_parts": self.active_temp_parts,
            "gpu_healthy": self.gpu_healthy,
            "anomaly_reason": self.anomaly_reason,
            "pcie_link_speed_sysfs": self.pcie_link_speed_sysfs,
            "pcie_link_width_sysfs": self.pcie_link_width_sysfs,
            "pcie_power_state_sysfs": self.pcie_power_state_sysfs,
            "aer_correctable_total": self.aer_correctable_total,
            "aer_nonfatal_total": self.aer_nonfatal_total,
            "aer_fatal_total": self.aer_fatal_total,
        }
        for field_name in NVIDIA_SMI_FIELDS:
            row[field_name] = self.nvidia_smi_values.get(field_name, "N/A")
        return row


def csv_fieldnames() -> list[str]:
    """Return the ordered list of CSV column headers for telemetry logs.

    Returns:
        List of column header strings.
    """
    base_fields = [
        "host_utc_iso",
        "active_variant",
        "completed_mpm_variants",
        "active_temp_parts",
        "gpu_healthy",
        "anomaly_reason",
        "pcie_link_speed_sysfs",
        "pcie_link_width_sysfs",
        "pcie_power_state_sysfs",
        "aer_correctable_total",
        "aer_nonfatal_total",
        "aer_fatal_total",
    ]
    return base_fields + list(NVIDIA_SMI_FIELDS)


def _read_text_file(path: Path, default: str = "N/A") -> str:
    """Read and strip a small sysfs or procfs file safely.

    Args:
        path: File path to read.
        default: Fallback string if the file cannot be read.

    Returns:
        Stripped file content or the fallback string.
    """
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return default


def parse_aer_counter_file(content: str) -> dict[str, int]:
    """Parse a Linux sysfs PCIe AER counter file into a dictionary.

    Args:
        content: Raw text content of an ``aer_dev_*`` sysfs file.

    Returns:
        Mapping of counter name to integer count.
    """
    counters: dict[str, int] = {}
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                counters[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return counters


def query_nvidia_smi() -> dict[str, str]:
    """Query ``nvidia-smi`` for all configured telemetry fields.

    Returns:
        Mapping from field name in ``NVIDIA_SMI_FIELDS`` to its string value.
    """
    query_arg = f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}"
    cmd = ["nvidia-smi", query_arg, "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {field: f"ERROR ({exc})" for field in NVIDIA_SMI_FIELDS}

    output_line = proc.stdout.strip().splitlines()
    if not output_line:
        err_msg = proc.stderr.strip() or f"exit_code={proc.returncode}"
        return {field: f"ERROR ({err_msg})" for field in NVIDIA_SMI_FIELDS}

    raw_values = [val.strip() for val in output_line[0].split(",")]
    result: dict[str, str] = {}
    for idx, field in enumerate(NVIDIA_SMI_FIELDS):
        result[field] = raw_values[idx] if idx < len(raw_values) else "N/A"
    return result


def inspect_experiment_progress(
    output_dir: Path, active_variant: str
) -> tuple[int, int]:
    """Count completed MPM variant files and active temporary batch files.

    Args:
        output_dir: Base output directory for Experiment 2 variants.
        active_variant: Name of the currently executing variant or model_variant.

    Returns:
        Tuple of ``(completed_mpm_variants, active_temp_parts)``.
    """
    mpm_dir = output_dir / "XAI_Maps" / "Multiscale_Pixel_Masking"
    gradcam_dir = output_dir / "XAI_Maps" / "GradCAM"
    if not mpm_dir.exists() and not gradcam_dir.exists():
        return 0, 0

    completed = (
        sum(
            1
            for model_name in ALL_EXP2_MODELS
            for variant in ALL_EXP1A_VARIANTS
            if (mpm_dir / f"{model_name}_{variant}_maps.npy").exists()
        )
        if mpm_dir.exists()
        else 0
    )

    temp_parts = 0
    for method_dir in (gradcam_dir, mpm_dir):
        if not method_dir.exists():
            continue
        specific_temp = method_dir / f"temp_{active_variant}"
        if specific_temp.is_dir():
            temp_parts += len(list(specific_temp.glob("part_*.npy")))
        elif active_variant in ALL_EXP1A_VARIANTS:
            clip_temp = method_dir / f"temp_clip_vit_b16_{active_variant}"
            if clip_temp.is_dir():
                temp_parts += len(list(clip_temp.glob("part_*.npy")))
        else:
            for temp_dir in method_dir.glob("temp_*"):
                if temp_dir.is_dir():
                    temp_parts += len(list(temp_dir.glob("part_*.npy")))

    return completed, temp_parts


def evaluate_health_and_anomalies(
    smi_values: dict[str, str],
    aer_nonfatal_total: int,
    aer_fatal_total: int,
) -> tuple[bool, str]:
    """Determine whether the GPU is healthy or experiencing an anomaly.

    Args:
        smi_values: Parsed ``nvidia-smi`` field values.
        aer_nonfatal_total: Total PCIe non-fatal AER count.
        aer_fatal_total: Total PCIe fatal AER count.

    Returns:
        Tuple of ``(gpu_healthy, anomaly_reason)``.
    """
    reasons: list[str] = []
    joined_values = " ".join(smi_values.values())

    if (
        "[GPU requires reset]" in joined_values
        or "ERR!" in joined_values
        or "ERROR (" in joined_values
    ):
        reasons.append("gpu_requires_reset_or_smi_error")

    recovery = smi_values.get("gpu_recovery_action", "None")
    if recovery not in ("None", "N/A", "[N/A]"):
        reasons.append(f"gpu_recovery_action={recovery}")

    if smi_values.get("clocks_event_reasons.hw_power_brake_slowdown") == "Active":
        reasons.append("hw_power_brake_slowdown_active")

    if smi_values.get("clocks_event_reasons.hw_thermal_slowdown") == "Active":
        reasons.append("hw_thermal_slowdown_active")

    if smi_values.get("clocks_event_reasons.hw_slowdown") == "Active":
        reasons.append("hw_slowdown_active")

    if aer_fatal_total > 0:
        reasons.append(f"pcie_aer_fatal={aer_fatal_total}")

    if aer_nonfatal_total > 0:
        reasons.append(f"pcie_aer_nonfatal={aer_nonfatal_total}")

    gpu_healthy = (
        "gpu_requires_reset_or_smi_error" not in reasons
        and f"gpu_recovery_action={recovery}" not in reasons
        and aer_fatal_total == 0
    )
    return gpu_healthy, ";".join(reasons) if reasons else "none"


def collect_telemetry_sample(
    active_variant: str = "idle",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    pci_device_path: Path = DEFAULT_PCI_DEVICE_PATH,
) -> TelemetrySample:
    """Collect a single GPU, PCIe, and experiment progress sample.

    Args:
        active_variant: Name of the variant currently running, or ``"idle"``.
        output_dir: Experiment 2 output directory.
        pci_device_path: Sysfs path for the GPU PCIe device.

    Returns:
        Populated ``TelemetrySample`` instance.
    """
    host_utc_iso = datetime.now(tz=UTC).isoformat()
    smi_values = query_nvidia_smi()

    aer_cor = parse_aer_counter_file(
        _read_text_file(pci_device_path / "aer_dev_correctable", "")
    )
    aer_nonfatal = parse_aer_counter_file(
        _read_text_file(pci_device_path / "aer_dev_nonfatal", "")
    )
    aer_fatal = parse_aer_counter_file(
        _read_text_file(pci_device_path / "aer_dev_fatal", "")
    )

    cor_total = aer_cor.get("TOTAL_ERR_COR", 0)
    nonfatal_total = aer_nonfatal.get("TOTAL_ERR_NONFATAL", 0)
    fatal_total = aer_fatal.get("TOTAL_ERR_FATAL", 0)

    gpu_healthy, anomaly_reason = evaluate_health_and_anomalies(
        smi_values, nonfatal_total, fatal_total
    )
    completed_mpm, active_parts = inspect_experiment_progress(
        output_dir, active_variant
    )

    return TelemetrySample(
        host_utc_iso=host_utc_iso,
        active_variant=active_variant,
        completed_mpm_variants=completed_mpm,
        active_temp_parts=active_parts,
        gpu_healthy=gpu_healthy,
        anomaly_reason=anomaly_reason,
        pcie_link_speed_sysfs=_read_text_file(pci_device_path / "current_link_speed"),
        pcie_link_width_sysfs=_read_text_file(pci_device_path / "current_link_width"),
        pcie_power_state_sysfs=_read_text_file(pci_device_path / "power_state"),
        aer_correctable_total=cor_total,
        aer_nonfatal_total=nonfatal_total,
        aer_fatal_total=fatal_total,
        nvidia_smi_values=smi_values,
    )


def write_crash_diagnostic_snapshot(
    telemetry_dir: Path,
    trigger_reason: str,
    recent_samples: list[TelemetrySample],
    pci_device_path: Path = DEFAULT_PCI_DEVICE_PATH,
    nvidia_proc_path: Path = DEFAULT_NVIDIA_PROC_PATH,
    extra_context: dict[str, str | int] | None = None,
) -> Path:
    """Dump a comprehensive JSON diagnostic file when an anomaly occurs.

    Args:
        telemetry_dir: Directory where diagnostic snapshots are saved.
        trigger_reason: Short description of why the snapshot was triggered.
        recent_samples: Rolling history of the most recent telemetry samples.
        pci_device_path: Sysfs path for the GPU PCIe device.
        nvidia_proc_path: Procfs path for the NVIDIA driver GPU entry.
        extra_context: Optional process or subprocess exit information.

    Returns:
        Path to the written JSON diagnostic file.
    """
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = telemetry_dir / f"crash_diagnostic_{ts_slug}.json"

    try:
        smi_q_proc = subprocess.run(
            [
                "nvidia-smi",
                "-q",
                "-d",
                "PERFORMANCE,POWER,TEMPERATURE,CLOCK,PIDS",
            ],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        smi_q_output = smi_q_proc.stdout.strip() or smi_q_proc.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        smi_q_output = f"Failed to run nvidia-smi -q: {exc}"

    payload = {
        "captured_at_utc": datetime.now(tz=UTC).isoformat(),
        "trigger_reason": trigger_reason,
        "extra_context": extra_context or {},
        "pcie_sysfs": {
            "current_link_speed": _read_text_file(
                pci_device_path / "current_link_speed"
            ),
            "current_link_width": _read_text_file(
                pci_device_path / "current_link_width"
            ),
            "max_link_speed": _read_text_file(pci_device_path / "max_link_speed"),
            "max_link_width": _read_text_file(pci_device_path / "max_link_width"),
            "power_state": _read_text_file(pci_device_path / "power_state"),
            "aer_dev_correctable": parse_aer_counter_file(
                _read_text_file(pci_device_path / "aer_dev_correctable", "")
            ),
            "aer_dev_nonfatal": parse_aer_counter_file(
                _read_text_file(pci_device_path / "aer_dev_nonfatal", "")
            ),
            "aer_dev_fatal": parse_aer_counter_file(
                _read_text_file(pci_device_path / "aer_dev_fatal", "")
            ),
        },
        "nvidia_procfs": {
            "information": _read_text_file(nvidia_proc_path / "information"),
            "power": _read_text_file(nvidia_proc_path / "power"),
        },
        "host_meminfo_head": "\n".join(
            _read_text_file(Path("/proc/meminfo")).splitlines()[:10]
        ),
        "recent_telemetry_history": [asdict(s) for s in recent_samples],
        "nvidia_smi_query_detail": smi_q_output,
    }

    with snapshot_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())

    logger.error("Wrote GPU crash diagnostic snapshot to %s", snapshot_path)
    return snapshot_path


def _parse_float(val: str | None) -> float | None:
    """Parse a float from an nvidia-smi or sysfs output string safely.

    Args:
        val: Raw string (e.g. '52.84', '36', 'N/A').

    Returns:
        Parsed float or None if parsing fails.
    """
    if val is None or val in ("N/A", "[N/A]", "None", ""):
        return None
    try:
        cleaned = val.split()[0].replace(",", "")
        return float(cleaned)
    except (ValueError, IndexError):
        return None


@dataclass
class VariantMetricsSummary:
    """Aggregated telemetry statistics for an active variant or execution phase."""

    variant: str
    sample_count: int = 0
    start_time_iso: str = ""
    end_time_iso: str = ""
    duration_sec: float = 0.0
    temp_min: float = float("inf")
    temp_max: float = float("-inf")
    temp_sum: float = 0.0
    power_min: float = float("inf")
    power_max: float = float("-inf")
    power_sum: float = 0.0
    vram_peak_mb: float = 0.0
    util_sum: float = 0.0
    util_max: float = 0.0
    anomaly_count: int = 0
    aer_fatal_count: int = 0

    @property
    def temp_avg(self) -> float:
        """Average GPU temperature across all recorded samples."""
        return self.temp_sum / self.sample_count if self.sample_count > 0 else 0.0

    @property
    def power_avg(self) -> float:
        """Average power draw across all recorded samples."""
        return self.power_sum / self.sample_count if self.sample_count > 0 else 0.0

    @property
    def util_avg(self) -> float:
        """Average GPU utilization across all recorded samples."""
        return self.util_sum / self.sample_count if self.sample_count > 0 else 0.0

    def update(self, sample: TelemetrySample) -> None:
        """Ingest a new telemetry sample into the aggregated statistics.

        Args:
            sample: Latest point-in-time snapshot.
        """
        self.sample_count += 1
        if not self.start_time_iso:
            self.start_time_iso = sample.host_utc_iso
        self.end_time_iso = sample.host_utc_iso

        temp = _parse_float(sample.nvidia_smi_values.get("temperature.gpu"))
        if temp is not None:
            self.temp_min = min(self.temp_min, temp)
            self.temp_max = max(self.temp_max, temp)
            self.temp_sum += temp

        power = _parse_float(sample.nvidia_smi_values.get("power.draw.instant"))
        if power is not None:
            self.power_min = min(self.power_min, power)
            self.power_max = max(self.power_max, power)
            self.power_sum += power

        vram = _parse_float(sample.nvidia_smi_values.get("memory.used"))
        if vram is not None:
            self.vram_peak_mb = max(self.vram_peak_mb, vram)

        util = _parse_float(sample.nvidia_smi_values.get("utilization.gpu"))
        if util is not None:
            self.util_max = max(self.util_max, util)
            self.util_sum += util

        if not sample.gpu_healthy or sample.anomaly_reason not in ("none", ""):
            self.anomaly_count += 1
        if sample.aer_fatal_total > 0:
            self.aer_fatal_count += sample.aer_fatal_total

    def to_dict(self) -> dict[str, str | int | float]:
        """Convert metrics summary into a JSON-serializable dictionary.

        Returns:
            Dictionary of metrics summary fields.
        """
        return {
            "variant": self.variant,
            "sample_count": self.sample_count,
            "start_time_iso": self.start_time_iso,
            "end_time_iso": self.end_time_iso,
            "duration_sec": round(self.duration_sec, 1),
            "temp_min": (
                round(self.temp_min, 1) if self.temp_min != float("inf") else 0.0
            ),
            "temp_max": (
                round(self.temp_max, 1) if self.temp_max != float("-inf") else 0.0
            ),
            "temp_avg": round(self.temp_avg, 1),
            "power_min": (
                round(self.power_min, 1) if self.power_min != float("inf") else 0.0
            ),
            "power_max": (
                round(self.power_max, 1) if self.power_max != float("-inf") else 0.0
            ),
            "power_avg": round(self.power_avg, 1),
            "vram_peak_mb": round(self.vram_peak_mb, 1),
            "util_avg": round(self.util_avg, 1),
            "util_max": round(self.util_max, 1),
            "anomaly_count": self.anomaly_count,
            "aer_fatal_count": self.aer_fatal_count,
        }


def format_summary_markdown(summaries: list[VariantMetricsSummary]) -> str:
    """Format variant metric summaries into a clean Markdown table.

    Args:
        summaries: Ordered list of variant telemetry summaries.

    Returns:
        Markdown-formatted string report.
    """
    lines = [
        "# GPU Telemetry Summary",
        "",
        f"Generated at: {datetime.now(tz=UTC).isoformat()}",
        "",
        (
            "| Variant | Duration | Samples | Avg Power | Peak Power | "
            "Avg Temp | Max Temp | Peak VRAM | Avg Util | Anomalies | Status |"
        ),
        (
            "| :--- | :--- | :--- | :--- | :--- | "
            ":--- | :--- | :--- | :--- | :--- | :--- |"
        ),
    ]
    for s in summaries:
        if s.sample_count == 0:
            continue
        dur_str = (
            f"{s.duration_sec / 60.0:.1f}m"
            if s.duration_sec > 0
            else f"{s.sample_count * 2.0 / 60.0:.1f}m"
        )
        t_avg = f"{s.temp_avg:.1f} °C" if s.temp_min != float("inf") else "N/A"
        t_max = f"{s.temp_max:.1f} °C" if s.temp_max != float("-inf") else "N/A"
        p_avg = f"{s.power_avg:.1f} W" if s.power_min != float("inf") else "N/A"
        p_max = f"{s.power_max:.1f} W" if s.power_max != float("-inf") else "N/A"
        v_peak = f"{s.vram_peak_mb:.0f} MiB" if s.vram_peak_mb > 0 else "N/A"
        u_avg = f"{s.util_avg:.1f} %"
        status = "OK" if s.anomaly_count == 0 and s.aer_fatal_count == 0 else "ANOMALY"

        lines.append(
            f"| {s.variant} | {dur_str} | {s.sample_count} | {p_avg} | {p_max} | "
            f"{t_avg} | {t_max} | {v_peak} | {u_avg} | {s.anomaly_count} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_summary_from_csv(csv_path: Path) -> list[VariantMetricsSummary]:
    """Parse an existing telemetry CSV file and aggregate per-variant statistics.

    Args:
        csv_path: Path to the ``gpu_telemetry.csv`` file.

    Returns:
        List of aggregated ``VariantMetricsSummary`` objects.
    """
    if not csv_path.exists():
        return []

    summaries: dict[str, VariantMetricsSummary] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            variant = row.get("active_variant", "unknown")
            if variant not in summaries:
                summaries[variant] = VariantMetricsSummary(variant=variant)
            summary = summaries[variant]
            summary.sample_count += 1
            if not summary.start_time_iso:
                summary.start_time_iso = row.get("host_utc_iso", "")
            summary.end_time_iso = row.get("host_utc_iso", "")

            temp = _parse_float(row.get("temperature.gpu"))
            if temp is not None:
                summary.temp_min = min(summary.temp_min, temp)
                summary.temp_max = max(summary.temp_max, temp)
                summary.temp_sum += temp

            power = _parse_float(row.get("power.draw.instant"))
            if power is not None:
                summary.power_min = min(summary.power_min, power)
                summary.power_max = max(summary.power_max, power)
                summary.power_sum += power

            vram = _parse_float(row.get("memory.used"))
            if vram is not None:
                summary.vram_peak_mb = max(summary.vram_peak_mb, vram)

            util = _parse_float(row.get("utilization.gpu"))
            if util is not None:
                summary.util_max = max(summary.util_max, util)
                summary.util_sum += util

            healthy = row.get("gpu_healthy", "True") == "True"
            anomaly = row.get("anomaly_reason", "none")
            if not healthy or anomaly not in ("none", ""):
                summary.anomaly_count += 1

            aer_fatal = int(_parse_float(row.get("aer_fatal_total")) or 0)
            if aer_fatal > 0:
                summary.aer_fatal_count += aer_fatal

    return list(summaries.values())


class BackgroundTelemetryMonitor:
    """Thread-based GPU telemetry collector with throttled CSV flushing and summaries."""

    def __init__(
        self,
        telemetry_dir: Path,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        interval_sec: float = 2.0,
        csv_interval_sec: float = 10.0,
        console_interval_sec: float = 30.0,
        history_size: int = 30,
    ) -> None:
        """Initialize the background telemetry monitor.

        Args:
            telemetry_dir: Directory for CSV logs and crash snapshots.
            output_dir: Experiment 2 output directory to watch for progress.
            interval_sec: Polling interval in seconds for health & crash detection.
            csv_interval_sec: Throttle interval in seconds between writing rows to
                the CSV file.
            console_interval_sec: Interval in seconds for emitting a readable live
                console status line (0 to disable).
            history_size: Number of recent samples retained in memory for
                crash snapshots.
        """
        self.telemetry_dir = telemetry_dir
        self.output_dir = output_dir
        self.interval_sec = interval_sec
        self.csv_interval_sec = max(interval_sec, csv_interval_sec)
        self.console_interval_sec = console_interval_sec
        self.csv_path = telemetry_dir / "gpu_telemetry.csv"
        self.summary_md_path = telemetry_dir / "gpu_telemetry_summary.md"
        self.summary_json_path = telemetry_dir / "gpu_telemetry_summary.json"
        self.recent_samples: deque[TelemetrySample] = deque(maxlen=history_size)
        self.variant_summaries: dict[str, VariantMetricsSummary] = {}
        self._active_variant = "idle"
        self._progress_status = ""
        self._variant_start_time = time.monotonic()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot_written_for_unhealthy = False

    def set_active_variant(self, variant_name: str) -> None:
        """Update active variant, finalize previous variant duration, and write summary.

        Args:
            variant_name: Variant identifier (e.g. ``"exp1a_variant5"``).
        """
        with self._lock:
            prev_variant = self._active_variant
            now_mono = time.monotonic()
            elapsed = now_mono - self._variant_start_time
            if prev_variant in self.variant_summaries:
                summary = self.variant_summaries[prev_variant]
                summary.duration_sec += elapsed
                if summary.sample_count > 0:
                    peak_pwr = (
                        summary.power_max if summary.power_max != float("-inf") else 0.0
                    )
                    max_temp = (
                        summary.temp_max if summary.temp_max != float("-inf") else 0.0
                    )
                    logger.info(
                        "Variant %s telemetry summary: duration=%.1fm, "
                        "avg_power=%.1fW (peak=%.1fW), avg_temp=%.1f°C (max=%.1f°C), anomalies=%d",
                        prev_variant,
                        summary.duration_sec / 60.0,
                        summary.power_avg,
                        peak_pwr,
                        summary.temp_avg,
                        max_temp,
                        summary.anomaly_count,
                    )
            self._active_variant = variant_name
            self._variant_start_time = now_mono
            self._save_summaries_locked()

    def set_progress_status(self, progress_status: str) -> None:
        """Update the overall progress label displayed in periodic telemetry logs.

        Args:
            progress_status: Short string (e.g. ``"12/70 (17%) | ETA: ~88h"``).
        """
        with self._lock:
            self._progress_status = progress_status

    def get_progress_status(self) -> str:
        """Return the current progress status string.

        Returns:
            Current progress status string.
        """
        with self._lock:
            return self._progress_status

    def get_active_variant(self) -> str:
        """Return the currently tracked variant label.

        Returns:
            Active variant string.
        """
        with self._lock:
            return self._active_variant

    def _save_summaries_locked(self) -> None:
        """Persist markdown and JSON summaries to disk while holding lock."""
        summaries = list(self.variant_summaries.values())
        if not summaries:
            return
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        md_text = format_summary_markdown(summaries)
        self.summary_md_path.write_text(md_text, encoding="utf-8")

        json_data = [s.to_dict() for s in summaries]
        with self.summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(json_data, handle, indent=2)

    def start(self) -> None:
        """Start the background telemetry polling thread."""
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._variant_start_time = time.monotonic()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="gpu-telemetry-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Started GPU telemetry monitor (poll=%.1fs, csv_flush=%.1fs, csv=%s)",
            self.interval_sec,
            self.csv_interval_sec,
            self.csv_path,
        )

    def stop(self) -> None:
        """Signal the background polling thread to stop and flush summaries."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.interval_sec + 5.0)
        with self._lock:
            now_mono = time.monotonic()
            elapsed = now_mono - self._variant_start_time
            if self._active_variant in self.variant_summaries:
                self.variant_summaries[self._active_variant].duration_sec += elapsed
            self._save_summaries_locked()
        logger.info(
            "Stopped GPU telemetry monitor. Summary written to %s",
            self.summary_md_path,
        )

    def _run_loop(self) -> None:
        """Poll telemetry at high frequency, throttle CSV writes, and log status."""
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames())
            if not file_exists:
                writer.writeheader()
                csv_file.flush()
                os.fsync(csv_file.fileno())

            last_csv_write_time = 0.0
            last_console_log_time = 0.0
            last_recorded_variant = ""

            while not self._stop_event.is_set():
                now_mono = time.monotonic()
                active_v = self.get_active_variant()
                sample = collect_telemetry_sample(
                    active_variant=active_v,
                    output_dir=self.output_dir,
                )
                self.recent_samples.append(sample)

                with self._lock:
                    if active_v not in self.variant_summaries:
                        self.variant_summaries[active_v] = VariantMetricsSummary(
                            variant=active_v
                        )
                    self.variant_summaries[active_v].update(sample)

                is_anomaly = not sample.gpu_healthy or sample.anomaly_reason not in (
                    "none",
                    "",
                )
                variant_changed = active_v != last_recorded_variant
                interval_elapsed = (
                    now_mono - last_csv_write_time
                ) >= self.csv_interval_sec

                if is_anomaly or variant_changed or interval_elapsed:
                    writer.writerow(sample.to_csv_row())
                    csv_file.flush()
                    os.fsync(csv_file.fileno())
                    last_csv_write_time = now_mono
                    last_recorded_variant = active_v

                if (
                    self.console_interval_sec > 0
                    and (now_mono - last_console_log_time) >= self.console_interval_sec
                ):
                    pwr = (
                        _parse_float(sample.nvidia_smi_values.get("power.draw.instant"))
                        or 0.0
                    )
                    temp = (
                        _parse_float(sample.nvidia_smi_values.get("temperature.gpu"))
                        or 0.0
                    )
                    vram = (
                        _parse_float(sample.nvidia_smi_values.get("memory.used")) or 0.0
                    )
                    util = (
                        _parse_float(sample.nvidia_smi_values.get("utilization.gpu"))
                        or 0.0
                    )
                    status_str = (
                        "OK"
                        if sample.gpu_healthy
                        else f"ANOMALY ({sample.anomaly_reason})"
                    )
                    prog_info = self.get_progress_status()
                    prog_prefix = f" [{prog_info}]" if prog_info else ""
                    logger.info(
                        "[TELEMETRY] [%s]%s Temp: %.0f°C | Power: %.1fW | VRAM: %.0f MiB | Util: %.0f%% | Status: %s",
                        active_v,
                        prog_prefix,
                        temp,
                        pwr,
                        vram,
                        util,
                        status_str,
                    )
                    last_console_log_time = now_mono

                if not sample.gpu_healthy and not self._snapshot_written_for_unhealthy:
                    self._snapshot_written_for_unhealthy = True
                    write_crash_diagnostic_snapshot(
                        telemetry_dir=self.telemetry_dir,
                        trigger_reason=sample.anomaly_reason,
                        recent_samples=list(self.recent_samples),
                    )
                elif sample.gpu_healthy and sample.anomaly_reason not in (
                    "none",
                    "",
                ):
                    logger.warning(
                        "GPU telemetry anomaly detected on %s: %s "
                        "(instant_power=%s W, avg_power=%s W, temp=%s C)",
                        sample.active_variant,
                        sample.anomaly_reason,
                        sample.nvidia_smi_values.get("power.draw.instant"),
                        sample.nvidia_smi_values.get("power.draw.average"),
                        sample.nvidia_smi_values.get("temperature.gpu"),
                    )

                self._stop_event.wait(self.interval_sec)


def _variant_maps_complete(
    output_dir: Path,
    model_name: str,
    variant: str,
    xai_methods: str,
) -> bool:
    """Check whether all requested XAI map files exist for a (model, variant) pair.

    Args:
        output_dir: Experiment 2 base output directory.
        model_name: Architecture identifier.
        variant: Variant tag (e.g. ``"exp1a_variant1"``).
        xai_methods: Requested XAI methods (``"gradcam"``, ``"mpm"``, or ``"both"``).

    Returns:
        True if all requested final ``.npy`` files exist on disk.
    """
    maps_dir = output_dir / "XAI_Maps"
    stem = f"{model_name}_{variant}_maps.npy"
    needs_gc = xai_methods in ("gradcam", "both")
    needs_mpm = xai_methods in ("mpm", "both")
    gc_ok = (maps_dir / "GradCAM" / stem).exists() if needs_gc else True
    mpm_ok = (
        (maps_dir / "Multiscale_Pixel_Masking" / stem).exists() if needs_mpm else True
    )
    return gc_ok and mpm_ok


def generate_progress_dashboard(
    output_dir: Path,
    target_models: list[str],
    variants: list[str],
    active_task: str,
    session_start_time: float,
    completed_durations_sec: list[float],
    xai_methods: str = "both",
) -> tuple[int, int, str]:
    """Build and save an auto-updating progress dashboard for long-running experiments.

    Args:
        output_dir: Experiment 2 base output directory.
        target_models: List of model architectures.
        variants: List of variant tags.
        active_task: Identifier of currently executing task (e.g. ``"vgg16_exp1a_variant3"``).
        session_start_time: Session start monotonic timestamp.
        completed_durations_sec: List of completed variant runtimes in seconds.
        xai_methods: Requested XAI methods.

    Returns:
        Tuple of ``(completed_count, total_count, short_status_string)``.
    """
    total_count = len(target_models) * len(variants)
    model_counts: dict[str, int] = {}
    model_status: dict[str, str] = {}

    for m in target_models:
        done_m = sum(
            1 for v in variants if _variant_maps_complete(output_dir, m, v, xai_methods)
        )
        model_counts[m] = done_m
        if done_m == len(variants):
            model_status[m] = "COMPLETED"
        elif any(f"{m}_{v}" == active_task for v in variants):
            model_status[m] = "IN_PROGRESS"
        elif done_m > 0:
            model_status[m] = "PARTIAL"
        else:
            model_status[m] = "PENDING"

    completed_count = sum(model_counts.values())
    pct = (completed_count / total_count * 100.0) if total_count > 0 else 100.0

    elapsed_sec = time.monotonic() - session_start_time
    elapsed_hr = elapsed_sec / 3600.0

    remaining_tasks = total_count - completed_count
    if completed_durations_sec:
        avg_sec = sum(completed_durations_sec) / len(completed_durations_sec)
        rem_sec = remaining_tasks * avg_sec
        rem_hr_str = f"~{rem_sec / 3600.0:.1f}h"
    else:
        rem_hr_str = "Estimating..."

    bar_len = 24
    filled_len = (
        int(bar_len * (completed_count / total_count)) if total_count > 0 else bar_len
    )
    bar_str = "[" + "█" * filled_len + "░" * (bar_len - filled_len) + "]"

    short_status = f"{completed_count}/{total_count} ({pct:.1f}%) | ETA: {rem_hr_str}"

    lines = [
        "# Experiment 2 Overall Execution Progress",
        "",
        f"- **Last Updated (UTC)**: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Current Task**: `{active_task}`",
        f"- **Overall Progress**: **{completed_count} / {total_count} variants** ({pct:.1f}%)",
        f"- **Progress Bar**: `{bar_str}`",
        f"- **Session Elapsed**: `{elapsed_hr:.2f} hours`",
        f"- **Estimated Time Remaining**: `{rem_hr_str}`",
        "",
        "## Architecture Status Breakdown",
        "",
        "| Architecture | Completed Variants | Total Variants | Status | Progress |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for m in target_models:
        cnt = model_counts[m]
        tot = len(variants)
        m_pct = (cnt / tot) * 100.0 if tot > 0 else 0.0
        m_filled = int(12 * (cnt / tot)) if tot > 0 else 12
        m_bar = "[" + "█" * m_filled + "░" * (12 - m_filled) + "]"
        lines.append(
            f"| **{m}** | {cnt} | {tot} | `{model_status[m]}` | `{m_bar}` {m_pct:.0f}% |"
        )

    lines.append("")
    progress_md_text = "\n".join(lines)
    progress_file = output_dir / "experiment_progress.md"
    progress_json = output_dir / "experiment_progress.json"
    try:
        progress_file.write_text(progress_md_text, encoding="utf-8")
        payload = {
            "last_updated_utc": datetime.now(tz=UTC).isoformat(),
            "active_task": active_task,
            "completed_variants": completed_count,
            "total_variants": total_count,
            "progress_percent": round(pct, 1),
            "session_elapsed_hours": round(elapsed_hr, 2),
            "estimated_remaining_hours": rem_hr_str,
            "model_breakdown": {
                m: {
                    "completed": model_counts[m],
                    "total": len(variants),
                    "status": model_status[m],
                }
                for m in target_models
            },
        }
        progress_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write progress dashboard: %s", exc)

    return completed_count, total_count, short_status


def run_isolated_experiment2_variants(
    variants: list[str],
    output_dir: Path,
    telemetry_dir: Path,
    models: list[str] | None = None,
    xai_methods: str = "both",
    interval_sec: float = 2.0,
    csv_interval_sec: float = 10.0,
    console_interval_sec: float = 30.0,
    variant_cooldown_sec: float = 15.0,
) -> int:
    """Run remaining Experiment 2 variants in isolated subprocesses across models.

    Starts the background telemetry monitor, verifies that the GPU is healthy,
    runs one fresh ``uv run python -m ...`` subprocess per incomplete
    ``(model, variant)`` pair so the CUDA context is completely destroyed
    between runs, and finally runs the cross-variant/cross-model/cross-method
    comparison pass with summary plot generation.

    Args:
        variants: Ordered list of variant names to ensure are completed.
        output_dir: Experiment 2 output directory.
        telemetry_dir: Directory for telemetry CSV and crash snapshots.
        models: Ordered list of model names to process (defaults to all 7 models).
        xai_methods: XAI methods to generate (``"gradcam"``, ``"mpm"``, or ``"both"``).
        interval_sec: Fast telemetry polling interval in seconds.
        csv_interval_sec: Throttle interval in seconds for CSV disk writes.
        console_interval_sec: Interval in seconds for live console status.
        variant_cooldown_sec: Pause between isolated variant subprocesses.

    Returns:
        Exit code (``0`` on full success, non-zero on GPU or subprocess error).
    """
    target_models = models if models is not None else list(ALL_EXP2_MODELS)
    initial_sample = collect_telemetry_sample(
        active_variant="preflight_check",
        output_dir=output_dir,
    )
    if not initial_sample.gpu_healthy:
        logger.error(
            "Pre-flight check failed: GPU is not healthy (%s). "
            "Writing diagnostic snapshot and aborting.",
            initial_sample.anomaly_reason,
        )
        write_crash_diagnostic_snapshot(
            telemetry_dir=telemetry_dir,
            trigger_reason=f"preflight_failed:{initial_sample.anomaly_reason}",
            recent_samples=[initial_sample],
        )
        return 2

    monitor = BackgroundTelemetryMonitor(
        telemetry_dir=telemetry_dir,
        output_dir=output_dir,
        interval_sec=interval_sec,
        csv_interval_sec=csv_interval_sec,
        console_interval_sec=console_interval_sec,
    )
    monitor.start()

    session_start_t = time.monotonic()
    completed_durations_sec: list[float] = []

    # Initialize dashboard
    _completed_count, total_count, short_status = generate_progress_dashboard(
        output_dir=output_dir,
        target_models=target_models,
        variants=variants,
        active_task="starting",
        session_start_time=session_start_t,
        completed_durations_sec=completed_durations_sec,
        xai_methods=xai_methods,
    )
    monitor.set_progress_status(short_status)
    logger.info("Overall experiment status at launch: %s", short_status)

    try:
        task_idx = 0
        for model_name in target_models:
            for variant in variants:
                task_idx += 1
                run_label = f"{model_name}_{variant}"
                if _variant_maps_complete(output_dir, model_name, variant, xai_methods):
                    logger.info(
                        "[%d/%d] Skipping %s (all requested %s maps already exist)",
                        task_idx,
                        total_count,
                        run_label,
                        xai_methods,
                    )
                    continue

                _completed_count, _total_count, short_status = (
                    generate_progress_dashboard(
                        output_dir=output_dir,
                        target_models=target_models,
                        variants=variants,
                        active_task=run_label,
                        session_start_time=session_start_t,
                        completed_durations_sec=completed_durations_sec,
                        xai_methods=xai_methods,
                    )
                )
                monitor.set_progress_status(short_status)
                monitor.set_active_variant(run_label)

                logger.info(
                    "\n%s\n[PROGRESS %d/%d] Overall: %s | Starting %s\nSession Elapsed: %.1fh\n%s",
                    "=" * 80,
                    task_idx,
                    total_count,
                    short_status,
                    run_label,
                    (time.monotonic() - session_start_t) / 3600.0,
                    "=" * 80,
                )

                cmd = [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "Image_Authenticity_prediction",
                    "experiment-two",
                    "--models",
                    model_name,
                    "--xai-methods",
                    xai_methods,
                    "--variants",
                    variant,
                ]
                start_t = time.monotonic()
                proc = subprocess.run(cmd, check=False)
                elapsed_sec = time.monotonic() - start_t
                elapsed_min = elapsed_sec / 60.0
                completed_durations_sec.append(elapsed_sec)

                post_sample = collect_telemetry_sample(
                    active_variant=run_label,
                    output_dir=output_dir,
                )
                maps_ok = _variant_maps_complete(
                    output_dir, model_name, variant, xai_methods
                )
                if proc.returncode != 0 or not post_sample.gpu_healthy or not maps_ok:
                    reason = (
                        f"subprocess_failed(rc={proc.returncode},"
                        f"maps_ok={maps_ok},"
                        f"anomaly={post_sample.anomaly_reason})"
                    )
                    logger.error(
                        "Variant %s failed after %.1f min: %s",
                        run_label,
                        elapsed_min,
                        reason,
                    )
                    write_crash_diagnostic_snapshot(
                        telemetry_dir=telemetry_dir,
                        trigger_reason=reason,
                        recent_samples=list(monitor.recent_samples),
                        extra_context={
                            "failed_variant": run_label,
                            "return_code": proc.returncode,
                            "maps_ok": int(maps_ok),
                            "elapsed_minutes": round(elapsed_min, 2),
                        },
                    )
                    return proc.returncode or 1

                _completed_count, _total_count, short_status = (
                    generate_progress_dashboard(
                        output_dir=output_dir,
                        target_models=target_models,
                        variants=variants,
                        active_task=f"{run_label}_cooldown",
                        session_start_time=session_start_t,
                        completed_durations_sec=completed_durations_sec,
                        xai_methods=xai_methods,
                    )
                )
                monitor.set_progress_status(short_status)

                logger.info(
                    "\n%s\n[COMPLETED %d/%d] Finished %s in %.1f min (Overall: %s)\nCooling down for %.0fs...\n%s",
                    "=" * 80,
                    task_idx,
                    total_count,
                    run_label,
                    elapsed_min,
                    short_status,
                    variant_cooldown_sec,
                    "=" * 80,
                )
                monitor.set_active_variant("cooldown")
                time.sleep(variant_cooldown_sec)

        monitor.set_active_variant("final_comparisons")
        compare_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "Image_Authenticity_prediction",
            "experiment-two",
            "--models",
            *target_models,
            "--xai-methods",
            xai_methods,
            "--comparison-only",
            "--save-plots",
            "--comparison-kinds",
            "within_model_variants",
            "between_model_architectures",
            "cross_methods",
            "--comparison-metrics",
            "correlation",
            "top_percent_iou_5",
            "top_percent_iou_15",
            "top_percent_iou_25",
        ]
        logger.info(
            "All variant maps present. Running final comparisons: %s",
            " ".join(compare_cmd),
        )
        comp_proc = subprocess.run(compare_cmd, check=False)
        return comp_proc.returncode
    finally:
        monitor.set_active_variant("stopped")
        monitor.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Returns:
        Configured ``ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Log flushed GPU/PCIe telemetry and optionally run Experiment 2 "
            "variants in isolated subprocesses."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Experiment 2 output directory.",
    )
    parser.add_argument(
        "--telemetry-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "Telemetry",
        help="Directory where telemetry logs and crash snapshots are saved.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds for health & crash detection (default: 2.0).",
    )
    parser.add_argument(
        "--csv-interval",
        type=float,
        default=10.0,
        help=(
            "Interval in seconds between writing rows to gpu_telemetry.csv "
            "(default: 10.0). Always writes immediately on anomalies or variant "
            "transitions."
        ),
    )
    parser.add_argument(
        "--console-interval",
        type=float,
        default=30.0,
        help=(
            "Interval in seconds for live console status one-liners "
            "(default: 30.0, set 0 to disable)."
        ),
    )
    parser.add_argument(
        "--run-exp2",
        action="store_true",
        help=(
            "Run remaining Experiment 2 variants in isolated subprocesses "
            "while recording telemetry in the background."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(ALL_EXP2_MODELS),
        choices=list(ALL_EXP2_MODELS),
        help="Model architectures to process when --run-exp2 is enabled.",
    )
    parser.add_argument(
        "--xai-methods",
        type=str,
        default="both",
        choices=["gradcam", "mpm", "both"],
        help="XAI methods to generate when --run-exp2 is enabled (default: both).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(ALL_EXP1A_VARIANTS),
        help="Variant identifiers to run when --run-exp2 is enabled.",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=15.0,
        help="Cooldown pause in seconds between isolated variant subprocesses.",
    )
    parser.add_argument(
        "--snapshot-now",
        action="store_true",
        help="Capture a single diagnostic snapshot immediately and exit.",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Display human-readable telemetry summary table from existing logs and exit.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display human-readable experiment progress dashboard and exit.",
    )
    return parser


def main() -> int:
    """Entry point for the GPU telemetry monitor and runner CLI.

    Returns:
        Process exit status code.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = build_arg_parser().parse_args()

    if args.show_summary:
        summary_md = args.telemetry_dir / "gpu_telemetry_summary.md"
        if summary_md.exists():
            import sys

            sys.stdout.write(summary_md.read_text(encoding="utf-8") + "\n")
            return 0
        csv_file = args.telemetry_dir / "gpu_telemetry.csv"
        if csv_file.exists():
            import sys

            summaries = generate_summary_from_csv(csv_file)
            sys.stdout.write(format_summary_markdown(summaries) + "\n")
            return 0
        logger.warning("No telemetry summary or CSV found in %s", args.telemetry_dir)
        return 1

    if args.show_progress:
        progress_md = args.output_dir / "experiment_progress.md"
        if progress_md.exists():
            import sys

            sys.stdout.write(progress_md.read_text(encoding="utf-8") + "\n")
            return 0
        logger.warning("No progress dashboard found in %s", args.output_dir)
        return 1

    if args.snapshot_now:
        sample = collect_telemetry_sample(
            active_variant="manual_snapshot",
            output_dir=args.output_dir,
        )
        write_crash_diagnostic_snapshot(
            telemetry_dir=args.telemetry_dir,
            trigger_reason=f"manual_snapshot:{sample.anomaly_reason}",
            recent_samples=[sample],
        )
        return 0

    if args.run_exp2:
        return run_isolated_experiment2_variants(
            variants=args.variants,
            output_dir=args.output_dir,
            telemetry_dir=args.telemetry_dir,
            models=args.models,
            xai_methods=args.xai_methods,
            interval_sec=args.interval,
            csv_interval_sec=args.csv_interval,
            console_interval_sec=args.console_interval,
            variant_cooldown_sec=args.cooldown_sec,
        )

    monitor = BackgroundTelemetryMonitor(
        telemetry_dir=args.telemetry_dir,
        output_dir=args.output_dir,
        interval_sec=args.interval,
        csv_interval_sec=args.csv_interval,
        console_interval_sec=args.console_interval,
    )
    monitor.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal; shutting down monitor.")
    finally:
        monitor.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
