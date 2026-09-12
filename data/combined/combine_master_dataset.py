"""Combine the matched electride registry and control dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ELECTRIDE_PATH = (
    PROJECT_ROOT
    / "data/electrides/final_dataset/inorganic_electride_registry_matched.csv"
)
CONTROL_PATH = PROJECT_ROOT / "data/control/final_dataset/control_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/combined"
OUTPUT_PATH = OUTPUT_DIR / "master_dataset.csv"


def load_dataset(path: Path, dataset_type: str, is_control: bool) -> pd.DataFrame:
    """Load one dataset and add its source metadata."""
    dataset = pd.read_csv(path)
    dataset["dataset_type"] = dataset_type
    dataset["is_control"] = is_control
    dataset["source_file"] = path.name
    return dataset


def main() -> None:
    """Create the combined master CSV."""
    if not ELECTRIDE_PATH.exists():
        raise FileNotFoundError(f"Matched electride CSV not found: {ELECTRIDE_PATH}")
    if not CONTROL_PATH.exists():
        raise FileNotFoundError(f"Control CSV not found: {CONTROL_PATH}")

    electride_df = load_dataset(ELECTRIDE_PATH, "matched_electride", False)
    control_df = load_dataset(CONTROL_PATH, "control", True)
    master_df = pd.concat([electride_df, control_df], ignore_index=True, sort=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(OUTPUT_PATH, index=False)

    duplicate_mp_ids = master_df["mp_id"].duplicated(keep=False).sum()
    print(f"Matched electride rows: {len(electride_df):,}")
    print(f"Control rows: {len(control_df):,}")
    print(f"Master rows: {len(master_df):,}")
    print(f"Master columns: {len(master_df.columns):,}")
    print(f"Rows belonging to duplicated MP IDs: {duplicate_mp_ids:,}")
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
