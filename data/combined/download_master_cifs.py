"""Download CIF files for every MP structure in the master dataset."""

from pathlib import Path

import os
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = PROJECT_ROOT / "data/combined/master_dataset.csv"
CIF_DIR = PROJECT_ROOT / "data/cifs"
LIGHTWEIGHT_PATH = PROJECT_ROOT / "data/combined/master_dataset_lightweight.csv"
QUERY_CHUNK_SIZE = 100


def download_structures(mp_ids: list[str]) -> tuple[dict[str, object], list[str]]:
    """Retrieve structures in batches and return them keyed by MP ID."""
    api_key = os.environ.get("MP_API_KEY", "").strip()
    if not api_key:
        raise ValueError("MP_API_KEY was not found in the project .env file.")

    structures = {}
    failed_ids = []
    fields = ["material_id", "structure"]

    with MPRester(api_key=api_key, mute_progress_bars=True) as mpr:
        for start in range(0, len(mp_ids), QUERY_CHUNK_SIZE):
            chunk = mp_ids[start : start + QUERY_CHUNK_SIZE]
            try:
                documents = mpr.materials.summary.search(
                    material_ids=chunk,
                    fields=fields,
                )
            except Exception as error:
                print(f"Could not retrieve MP IDs {start}:{start + len(chunk)}: {error}")
                failed_ids.extend(chunk)
                continue

            returned_ids = set()
            for document in documents:
                mp_id = str(document.material_id)
                returned_ids.add(mp_id)
                if document.structure is not None:
                    structures[mp_id] = document.structure

            failed_ids.extend(mp_id for mp_id in chunk if mp_id not in returned_ids)
            print(f"Retrieved {min(start + len(chunk), len(mp_ids))}/{len(mp_ids)} MP IDs")

    return structures, sorted(set(failed_ids))


def main() -> None:
    """Download CIFs, update the master CSV, and write a lightweight CSV."""
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_PATH}")

    load_dotenv(PROJECT_ROOT / ".env")
    master_df = pd.read_csv(MASTER_PATH)
    mp_ids = sorted(master_df["mp_id"].dropna().astype(str).unique())
    CIF_DIR.mkdir(parents=True, exist_ok=True)

    structures, failed_ids = download_structures(mp_ids)
    cif_paths = {}
    for mp_id, structure in structures.items():
        cif_path = CIF_DIR / f"{mp_id}.cif"
        CifWriter(structure).write_file(cif_path)
        cif_paths[mp_id] = str(cif_path.relative_to(PROJECT_ROOT))

    master_df["cif_path"] = master_df["mp_id"].astype(str).map(cif_paths)
    master_df.to_csv(MASTER_PATH, index=False)

    lightweight_df = pd.DataFrame({
        "normalized_formula": master_df["normalized_formula"].fillna(master_df["formula"]),
        "material_id": master_df["material_id"].fillna(master_df["mp_id"]),
        "mp_id": master_df["mp_id"],
        "electride_dimensionality": master_df["electride_dimensionality"].fillna("not_reported"),
        "cif_path": master_df["cif_path"].fillna("not_found"),
        "space_group": master_df["space_group"],
        "pressure": master_df["pressure"].fillna("not_reported"),
    })
    lightweight_df.to_csv(LIGHTWEIGHT_PATH, index=False)

    print(f"Master rows: {len(master_df):,}")
    print(f"Unique MP IDs: {len(mp_ids):,}")
    print(f"CIF files written: {len(cif_paths):,}")
    print(f"MP IDs without CIFs: {len(failed_ids):,}")
    if failed_ids:
        print("Missing MP IDs:", ", ".join(failed_ids))
    print(f"Updated: {MASTER_PATH}")
    print(f"Wrote: {LIGHTWEIGHT_PATH}")


if __name__ == "__main__":
    main()
