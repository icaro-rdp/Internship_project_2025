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
        active_variant: Name of the currently executing variant.

    Returns:
        Tuple of ``(completed_mpm_variants, active_temp_parts)``.
    """
    mpm_dir = output_dir / "XAI_Maps" / "Multiscale_Pixel_Masking"
    if not mpm_dir.exists():
        return 0, 0

    completed = sum(
        1
        for variant in ALL_EXP1A_VARIANTS
        if (mpm_dir / f"clip_vit_b16_{variant}_maps.npy").exists()
    )

    temp_parts = 0
    if active_variant in ALL_EXP1A_VARIANTS:
        temp_dir = mpm_dir / f"temp_clip_vit_b16_{active_variant}"
        if temp_dir.exists():
            temp_parts = len(list(temp_dir.glob("part_*.npy")))
    else:
        for temp_dir in mpm_dir.glob("temp_clip_vit_b16_*"):
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


class BackgroundTelemetryMonitor:
    """Thread-based GPU telemetry collector that flushes every sample to CSV."""

    def __init__(
        self,
        telemetry_dir: Path,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        interval_sec: float = 2.0,
        history_size: int = 30,
    ) -> None:
        """Initialize the background telemetry monitor.

        Args:
            telemetry_dir: Directory for CSV logs and crash snapshots.
            output_dir: Experiment 2 output directory to watch for progress.
            interval_sec: Polling interval in seconds.
            history_size: Number of recent samples retained in memory for
                crash snapshots.
        """
        self.telemetry_dir = telemetry_dir
        self.output_dir = output_dir
        self.interval_sec = interval_sec
        self.csv_path = telemetry_dir / "gpu_telemetry.csv"
        self.recent_samples: deque[TelemetrySample] = deque(maxlen=history_size)
        self._active_variant = "idle"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot_written_for_unhealthy = False

    def set_active_variant(self, variant_name: str) -> None:
        """Update the label of the currently executing variant.

        Args:
            variant_name: Variant identifier (e.g. ``"exp1a_variant5"``).
        """
        with self._lock:
            self._active_variant = variant_name

    def get_active_variant(self) -> str:
        """Return the currently tracked variant label.

        Returns:
            Active variant string.
        """
        with self._lock:
            return self._active_variant

    def start(self) -> None:
        """Start the background telemetry polling thread."""
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="gpu-telemetry-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Started GPU telemetry monitor (interval=%.1fs, csv=%s)",
            self.interval_sec,
            self.csv_path,
        )

    def stop(self) -> None:
        """Signal the background polling thread to stop and wait for exit."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.interval_sec + 5.0)
        logger.info("Stopped GPU telemetry monitor.")

    def _run_loop(self) -> None:
        """Poll telemetry continuously and fsync every row to disk."""
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames())
            if not file_exists:
                writer.writeheader()
                csv_file.flush()
                os.fsync(csv_file.fileno())

            while not self._stop_event.is_set():
                sample = collect_telemetry_sample(
                    active_variant=self.get_active_variant(),
                    output_dir=self.output_dir,
                )
                self.recent_samples.append(sample)
                writer.writerow(sample.to_csv_row())
                csv_file.flush()
                os.fsync(csv_file.fileno())

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


def run_isolated_experiment2_variants(
    variants: list[str],
    output_dir: Path,
    telemetry_dir: Path,
    interval_sec: float = 2.0,
    variant_cooldown_sec: float = 15.0,
) -> int:
    """Run remaining Experiment 2 MPM variants in isolated subprocesses.

    Starts the background telemetry monitor, verifies that the GPU is healthy,
    runs one fresh ``uv run python -m ...`` subprocess per incomplete variant
    so the CUDA context is completely destroyed between variants, and finally
    runs the cross-variant/cross-method comparison pass.

    Args:
        variants: Ordered list of variant names to ensure are completed.
        output_dir: Experiment 2 output directory.
        telemetry_dir: Directory for telemetry CSV and crash snapshots.
        interval_sec: Telemetry sampling interval in seconds.
        variant_cooldown_sec: Pause between isolated variant subprocesses.

    Returns:
        Exit code (``0`` on full success, non-zero on GPU or subprocess error).
    """
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
    )
    monitor.start()

    try:
        mpm_dir = output_dir / "XAI_Maps" / "Multiscale_Pixel_Masking"
        for variant in variants:
            target_file = mpm_dir / f"clip_vit_b16_{variant}_maps.npy"
            if target_file.exists():
                logger.info("Skipping %s (already exists at %s)", variant, target_file)
                continue

            monitor.set_active_variant(variant)
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "Image_Authenticity_prediction.main.Experiments.experiment_two",
                "--variants",
                variant,
                "--methods",
                "Multiscale_Pixel_Masking",
                "--output_dir",
                str(output_dir),
            ]
            logger.info(
                "Launching isolated subprocess for %s: %s",
                variant,
                " ".join(cmd),
            )
            start_t = time.monotonic()
            proc = subprocess.run(cmd, check=False)
            elapsed_min = (time.monotonic() - start_t) / 60.0

            post_sample = collect_telemetry_sample(
                active_variant=variant,
                output_dir=output_dir,
            )
            if proc.returncode != 0 or not post_sample.gpu_healthy:
                reason = (
                    f"subprocess_failed(rc={proc.returncode},"
                    f"anomaly={post_sample.anomaly_reason})"
                )
                logger.error(
                    "Variant %s failed after %.1f min: %s",
                    variant,
                    elapsed_min,
                    reason,
                )
                write_crash_diagnostic_snapshot(
                    telemetry_dir=telemetry_dir,
                    trigger_reason=reason,
                    recent_samples=list(monitor.recent_samples),
                    extra_context={
                        "failed_variant": variant,
                        "return_code": proc.returncode,
                        "elapsed_minutes": round(elapsed_min, 2),
                    },
                )
                return proc.returncode or 1

            logger.info(
                "Completed %s in %.1f min. Cooling down for %.0f s...",
                variant,
                elapsed_min,
                variant_cooldown_sec,
            )
            monitor.set_active_variant("cooldown")
            time.sleep(variant_cooldown_sec)

        monitor.set_active_variant("final_comparisons")
        compare_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "Image_Authenticity_prediction.main.Experiments.experiment_two",
            "--variants",
            "exp1a",
            "--methods",
            "GradCAM",
            "Multiscale_Pixel_Masking",
            "--output_dir",
            str(output_dir),
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
        help="Directory where gpu_telemetry.csv and crash snapshots are saved.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--run-exp2",
        action="store_true",
        help=(
            "Run remaining Experiment 2 MPM variants in isolated subprocesses "
            "while recording telemetry in the background."
        ),
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
            interval_sec=args.interval,
            variant_cooldown_sec=args.cooldown_sec,
        )

    monitor = BackgroundTelemetryMonitor(
        telemetry_dir=args.telemetry_dir,
        output_dir=args.output_dir,
        interval_sec=args.interval,
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
