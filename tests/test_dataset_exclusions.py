"""Unit tests for dataset exclusions, prompt taxonomy, and manifest consistency."""

from pathlib import Path

import pandas as pd

CANONICAL_DATASET_DIR = (
    Path(__file__).resolve().parent.parent
    / "Image_Authenticity_prediction"
    / "Dataset"
    / "AIGCIQA2023"
)


def test_real_images_annotations_existence_and_counts() -> None:
    """Verify real_images_annotations.csv has exactly 1,368 images across 6 models."""
    csv_path = CANONICAL_DATASET_DIR / "real_images_annotations.csv"
    assert csv_path.exists(), f"Expected annotations CSV at {csv_path}"

    df_kept = pd.read_csv(csv_path)
    expected_cols = [
        "Quality",
        "Authenticity",
        "Correspondence",
        "image_path",
        "model",
        "Challenge",
        "Category",
        "Prompt",
    ]
    assert list(df_kept.columns) == expected_cols
    assert len(df_kept) == 1368, f"Expected 1,368 kept images, found {len(df_kept)}"

    # Exactly 57 unique prompts with 24 images each
    prompt_counts = df_kept.groupby(df_kept["Prompt"].str.strip()).size()
    assert len(prompt_counts) == 57, (
        f"Expected 57 unique prompts, found {len(prompt_counts)}"
    )
    assert (prompt_counts == 24).all(), "Every kept prompt must have exactly 24 images"

    # Exactly 228 images per model across 6 models
    model_counts = df_kept["model"].value_counts()
    assert len(model_counts) == 6, (
        f"Expected 6 generative models, found {len(model_counts)}"
    )
    assert (model_counts == 228).all(), (
        "Every generative model must have 228 kept images"
    )


def test_prompt_exclusion_manifest_taxonomy() -> None:
    """Verify prompt_exclusion_manifest.csv partitions all 100 prompts into 57 kept and 43 removed."""
    manifest_path = CANONICAL_DATASET_DIR / "prompt_exclusion_manifest.csv"
    assert manifest_path.exists(), f"Expected manifest file at {manifest_path}"

    df_manifest = pd.read_csv(manifest_path)
    assert len(df_manifest) == 100, (
        f"Expected 100 total prompts, found {len(df_manifest)}"
    )

    status_counts = df_manifest["Status"].value_counts().to_dict()
    assert status_counts.get("Kept") == 57, (
        f"Expected 57 kept prompts, found {status_counts.get('Kept')}"
    )
    assert status_counts.get("Removed") == 43, (
        f"Expected 43 removed prompts, found {status_counts.get('Removed')}"
    )

    # Verify Challenge-level exclusions (Imagination, Style & Format, Writing & Symbols all removed)
    for challenge in ["Imagination", "Style & Format", "Writing & Symbols"]:
        challenge_subset = df_manifest[df_manifest["Challenge"] == challenge]
        assert len(challenge_subset) == 10, f"Expected 10 prompts for {challenge}"
        assert (challenge_subset["Status"] == "Removed").all(), (
            f"All prompts in {challenge} must be Removed"
        )

    # Verify Basic challenge is completely kept (10/10)
    basic_subset = df_manifest[df_manifest["Challenge"] == "Basic"]
    assert len(basic_subset) == 10
    assert (basic_subset["Status"] == "Kept").all(), (
        "All prompts in Basic challenge must be Kept"
    )


def test_removed_images_details_counts_and_divisibility() -> None:
    """Verify removed_images_details.csv contains exactly 1,032 rows and is divisible by 24."""
    details_path = CANONICAL_DATASET_DIR / "removed_images_details.csv"
    assert details_path.exists(), f"Expected removed images details at {details_path}"

    df_removed = pd.read_csv(details_path)
    assert len(df_removed) == 1032, (
        f"Expected 1,032 removed images, found {len(df_removed)}"
    )
    assert len(df_removed) % 24 == 0, (
        "Excluded image count must be evenly divisible by 24"
    )

    # Exactly 43 unique prompts with 24 images each
    prompt_counts = df_removed.groupby(df_removed["Prompt"].str.strip()).size()
    assert len(prompt_counts) == 43, (
        f"Expected 43 removed prompts, found {len(prompt_counts)}"
    )
    assert (prompt_counts == 24).all(), (
        "Every removed prompt must have exactly 24 images"
    )

    # Exactly 172 images per model across 6 models
    model_counts = df_removed["model"].value_counts()
    assert len(model_counts) == 6
    assert (model_counts == 172).all(), (
        "Every generative model must have 172 removed images"
    )


def test_partition_disjointness_and_completeness() -> None:
    """Verify kept and removed image sets are disjoint and sum to 2,400 images."""
    kept_path = CANONICAL_DATASET_DIR / "real_images_annotations.csv"
    removed_path = CANONICAL_DATASET_DIR / "removed_images_details.csv"

    assert kept_path.exists(), f"real_images_annotations.csv not found at {kept_path}"
    assert removed_path.exists(), (
        f"removed_images_details.csv not found at {removed_path}"
    )

    df_kept = pd.read_csv(kept_path)
    df_removed = pd.read_csv(removed_path)

    kept_paths = set(df_kept["image_path"])
    removed_paths = set(df_removed["image_path"])

    assert len(kept_paths & removed_paths) == 0, (
        "Kept and removed image sets must have 0 overlap"
    )
    assert len(kept_paths | removed_paths) == 2400, (
        "Union of kept and removed must equal 2,400 images"
    )
