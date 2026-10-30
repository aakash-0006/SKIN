"""Dataset discovery, validation and lesion-level splitting for HAM10000."""
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMG_EXT = {".jpg", ".jpeg", ".png"}


def load_config():
    with open(PROJECT_DIR / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    # Environment variables make external datasets portable across machines.
    cfg["dataset_root"] = os.environ.get("HAM_DATASET_ROOT", cfg["dataset_root"])
    cfg["isic_dataset_root"] = os.environ.get("ISIC_DATASET_ROOT", cfg.get("isic_dataset_root", ""))
    return cfg


def find_metadata(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    hits = list(root.rglob("HAM10000_metadata.csv"))
    if not hits:
        raise FileNotFoundError(f"HAM10000_metadata.csv not found anywhere under {root}")
    return hits[0]


def index_images(root):
    """Map image_id (file stem) -> full path, scanning all nested folders."""
    mapping, dupes = {}, []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            if p.stem in mapping:
                dupes.append(str(p))
            else:
                mapping[p.stem] = str(p)
    return mapping, dupes


def resolve_split_paths(df, root):
    """Keep saved split membership but remap image paths when a dataset moved."""
    out = df.copy()
    files, _ = index_images(root)
    paths = []
    missing_ids = []
    for row in out.itertuples(index=False):
        current = Path(getattr(row, "path"))
        image_id = str(getattr(row, "image_id"))
        if current.is_file():
            paths.append(str(current))
        elif image_id in files:
            paths.append(str(files[image_id]))
        else:
            missing_ids.append(image_id)
            paths.append(str(current))
    out["path"] = paths
    if missing_ids:
        raise FileNotFoundError(f"{len(missing_ids)} saved split images are missing under {root}; sample: {missing_ids[:10]}")
    return out


def build_dataframe(root, report_path=None):
    meta_path = find_metadata(root)
    df = pd.read_csv(meta_path)
    for col in ("image_id", "lesion_id", "dx"):
        if col not in df.columns:
            raise ValueError(f"Metadata is missing required column '{col}'. Columns: {list(df.columns)}")

    n_rows = len(df)
    dup_rows = int(df.duplicated("image_id").sum())
    df = df.drop_duplicates("image_id").copy()

    files, dup_files = index_images(root)
    df["path"] = df["image_id"].map(files)
    missing = df[df["path"].isna()]["image_id"].tolist()
    orphans = sorted(set(files) - set(df["image_id"]))
    df = df.dropna(subset=["path"]).reset_index(drop=True)

    unknown = sorted(set(df["dx"]) - set(CLASSES))
    mixed_lesions = int((df.groupby("lesion_id")["dx"].nunique() > 1).sum())

    report = {
        "metadata_file": str(meta_path),
        "metadata_rows": n_rows,
        "duplicate_image_id_rows_removed": dup_rows,
        "image_files_found": len(files),
        "duplicate_filenames_in_different_folders": len(dup_files),
        "metadata_rows_without_image": len(missing),
        "missing_image_ids_sample": missing[:20],
        "image_files_not_in_metadata": len(orphans),
        "unknown_class_labels": unknown,
        "lesions_with_conflicting_labels": mixed_lesions,
        "unique_lesions": int(df["lesion_id"].nunique()),
        "usable_images": len(df),
        "class_counts": df["dx"].value_counts().to_dict(),
    }
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    if len(df) == 0:
        raise RuntimeError("No usable images: metadata IDs did not match any image files.\n" + json.dumps(report, indent=2))
    if unknown:
        raise RuntimeError(f"Unexpected class labels in metadata: {unknown}")
    return df, report


def split_by_lesion(df, seed=42, train=0.70, val=0.15, min_per_class=5):
    """Group split by lesion_id so one lesion never spans partitions."""
    for attempt in range(100):
        s = seed + attempt
        gss = GroupShuffleSplit(n_splits=1, train_size=train, random_state=s)
        tr_idx, rest_idx = next(gss.split(df, groups=df["lesion_id"]))
        rest = df.iloc[rest_idx]
        gss2 = GroupShuffleSplit(n_splits=1, train_size=val / (1 - train), random_state=s)
        v_idx, t_idx = next(gss2.split(rest, groups=rest["lesion_id"]))
        out = df.copy()
        out["split"] = ""
        out.loc[df.index[tr_idx], "split"] = "train"
        out.loc[rest.index[v_idx], "split"] = "val"
        out.loc[rest.index[t_idx], "split"] = "test"
        counts = out.groupby(["split", "dx"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
        if (counts.values >= min_per_class).all():
            sets = {k: set(g["lesion_id"]) for k, g in out.groupby("split")}
            assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]), "lesion leakage"
            return out, counts
    raise RuntimeError("Could not find a split with enough samples of every class in each partition.")
