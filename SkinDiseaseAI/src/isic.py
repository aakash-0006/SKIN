"""ISIC 2019 discovery, explicit label mapping, and integrity inspection.

The accepted mappings reflect the ISIC 2019 one-hot diagnosis column names:
MEL/NV/BCC/AK/BKL/DF/VASC. SCC and UNK have no compatible HAM10000 class.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from src.dataset import CLASSES, PROJECT_DIR, resolve_split_paths

ImageFile.LOAD_TRUNCATED_IMAGES = False

# The columns are source diagnoses in ISIC_2019_Training_GroundTruth.csv.
ISIC_TO_HAM = {
    "MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec",
    "BKL": "bkl", "DF": "df", "VASC": "vasc",
    "SCC": None, "UNK": None,
}
HAM_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def _resolve(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_DIR / p).resolve()


def _find_file(root, name):
    p = Path(root) / name
    if p.is_file():
        return p
    found = next(Path(root).rglob(name), None)
    if found is None:
        raise FileNotFoundError(f"Required file {name!r} was not found under {root}")
    return found


def _phash(image):
    """64-bit DCT perceptual hash; only used to flag conservative review candidates."""
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    x = np.arange(32, dtype=np.float32)
    k = np.arange(8, dtype=np.float32)[:, None]
    basis = np.cos((np.pi / 32) * (x + 0.5) * k)
    basis[0] *= 1 / np.sqrt(2)
    basis *= np.sqrt(2 / 32)
    dct = basis @ gray @ basis.T
    low = dct[:8, :8].reshape(-1)
    med = np.median(low[1:])
    bits = low > med
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _image_index(root):
    by_id, duplicate_ids = {}, []
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXT:
            if path.stem in by_id:
                duplicate_ids.append({"image_id": path.stem, "path": str(path)})
            else:
                by_id[path.stem] = path
    return by_id, duplicate_ids


def _canonical(value):
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        items = [_canonical(v) for v in value]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True, default=str))
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_labels(path):
    raw = pd.read_csv(path)
    if "image" not in raw:
        raise ValueError(f"ISIC ground truth is missing 'image': {list(raw.columns)}")
    diagnosis_columns = [c for c in raw.columns if c != "image"]
    raw.attrs["unmapped_source_columns"] = sorted(set(diagnosis_columns) - set(ISIC_TO_HAM))
    for c in diagnosis_columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    labels, status = [], []
    for row in raw[diagnosis_columns].itertuples(index=False, name=None):
        positives = [diagnosis_columns[i] for i, val in enumerate(row) if val == 1]
        invalid = any(pd.isna(val) or val not in (0, 1) for val in row)
        if invalid or len(positives) != 1:
            labels.append(None)
            status.append("invalid_one_hot")
        else:
            source_dx = positives[0]
            labels.append(ISIC_TO_HAM[source_dx])
            status.append("compatible" if ISIC_TO_HAM[source_dx] else "incompatible")
    raw["source_dx"] = [next((c for c, val in zip(diagnosis_columns, row) if val == 1), "unlabeled")
                         for row in raw[diagnosis_columns].itertuples(index=False, name=None)]
    raw["dx"] = labels
    raw["label_status"] = status
    return raw


def inspect_isic(isic_root, ham_root, splits_csv, report_path, near_threshold=4):
    """Inspect metadata and all image files; no files are modified."""
    if not isinstance(near_threshold, int) or near_threshold < 0 or near_threshold > 6:
        raise ValueError("near duplicate Hamming threshold must be an integer from 0 through 6")
    root = Path(isic_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ISIC dataset folder does not exist: {root}")
    gt_path = _find_file(root, "ISIC_2019_Training_GroundTruth.csv")
    meta_path = _find_file(root, "ISIC_2019_Training_Metadata.csv")
    labels = _read_labels(gt_path)
    unmapped_source_columns = labels.attrs.get("unmapped_source_columns", [])
    source_diagnosis_columns = [c for c in labels.columns if c not in {"image", "source_dx", "dx", "label_status"}]
    metadata = pd.read_csv(meta_path)
    if "image" not in metadata:
        raise ValueError("ISIC metadata is missing the 'image' ID column")
    if metadata["image"].duplicated().any():
        raise ValueError("ISIC metadata contains duplicate image IDs; resolve before training")
    image_map, duplicate_names = _image_index(root)
    labels["path"] = labels["image"].map(lambda i: str(image_map[i]) if i in image_map else None)
    labels = labels.merge(metadata, on="image", how="left", suffixes=("", "_metadata"))

    missing_images = labels[labels.path.isna()]["image"].astype(str).tolist()
    gt_without_metadata = sorted(set(labels.image.astype(str)) - set(metadata.image.astype(str)))
    metadata_without_gt = sorted(set(metadata.image.astype(str)) - set(labels.image.astype(str)))
    invalid, hashes, phashes = [], {}, {}
    valid = set()
    for image_index, (image_id, path) in enumerate(image_map.items(), start=1):
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
                phashes[image_id] = _phash(im)
            with open(path, "rb") as f:
                hashes[image_id] = hashlib.sha256(f.read()).hexdigest()
            valid.add(image_id)
        except Exception as exc:
            invalid.append({"image_id": image_id, "path": str(path), "error": str(exc)[:240]})
        if image_index % 1000 == 0:
            print(f"Checked ISIC images: {image_index}/{len(image_map)}")
    labels["image_valid"] = labels.image.astype(str).isin(valid)

    ham_root = Path(ham_root).expanduser().resolve()
    ham_map, ham_dupes = _image_index(ham_root)
    ham_metadata_path = _find_file(ham_root, "HAM10000_metadata.csv")
    ham_metadata = pd.read_csv(ham_metadata_path)
    if not {"image_id", "dx"}.issubset(ham_metadata.columns):
        raise ValueError(f"HAM metadata needs image_id and dx columns: {list(ham_metadata.columns)}")
    shared_image_ids = sorted(set(image_map) & set(ham_map))
    split_path = Path(splits_csv).expanduser().resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"Existing split manifest not found: {split_path}")
    splits = pd.read_csv(split_path)
    required = {"image_id", "lesion_id", "dx", "path", "split"}
    if not required.issubset(splits.columns):
        raise ValueError(f"Existing split manifest lacks columns: {sorted(required - set(splits.columns))}")
    splits = resolve_split_paths(splits, ham_root)
    test_ids = set(splits.loc[splits.split == "test", "image_id"].astype(str))
    # Exact hashes are computed for every ISIC image above and each HAM image
    # that participates in the preserved train/validation/test manifest.
    ham_records = []
    for row_index, row in enumerate(splits.itertuples(index=False), start=1):
        p = Path(row.path)
        if not p.is_file():
            ham_records.append({"image_id": str(row.image_id), "path": str(p), "split": str(row.split),
                                "sha256": None, "phash": None, "error": "file does not exist"})
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                h = _phash(im)
            with open(p, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            ham_records.append({"image_id": str(row.image_id), "path": str(p), "split": str(row.split),
                                "sha256": sha, "phash": h})
        except Exception as exc:
            # HAM corruption is reported; the existing test rows are never moved.
            ham_records.append({"image_id": str(row.image_id), "path": str(p), "split": str(row.split),
                                "sha256": None, "phash": None, "error": str(exc)[:240]})
        if row_index % 1000 == 0:
            print(f"Checked HAM split images: {row_index}/{len(splits)}")

    ham_exact_internal = []
    ham_sha = {}
    for rec in ham_records:
        if rec.get("sha256"):
            ham_sha.setdefault(rec["sha256"], []).append(rec)
    for same in ham_sha.values():
        if len(same) > 1:
            for i, left in enumerate(same):
                for right in same[i + 1:]:
                    if left["split"] != right["split"]:
                        ham_exact_internal.append({"image_a": left["image_id"], "split_a": left["split"],
                                                   "image_b": right["image_id"], "split_b": right["split"],
                                                   "sha256": left["sha256"]})

    isic_records = [{"image_id": i, "path": str(p), "sha256": hashes.get(i), "phash": phashes.get(i)}
                    for i, p in image_map.items() if i in valid]
    isic_by_sha = {}
    for rec in isic_records:
        isic_by_sha.setdefault(rec["sha256"], []).append(rec)
    exact_internal = []
    for same in isic_by_sha.values():
        if len(same) > 1:
            for i, left in enumerate(same):
                for right in same[i + 1:]:
                    exact_internal.append({"image_a": left["image_id"], "image_b": right["image_id"],
                                           "sha256": left["sha256"]})
    ham_by_sha = {}
    for rec in ham_records:
        if rec.get("sha256"):
            ham_by_sha.setdefault(rec["sha256"], []).append(rec)
    exact_pairs = []
    for rec in isic_records:
        for old in ham_by_sha.get(rec["sha256"], []):
            exact_pairs.append({"isic_image": rec["image_id"], "ham_image": old["image_id"],
                                "ham_split": old["split"], "sha256": rec["sha256"]})

    # If two 64-bit hashes differ in at most t bits, at least one of t+1
    # disjoint segments must match exactly. Segment buckets give a complete,
    # substantially faster candidate search than enumerating all bit masks.
    segment_count = near_threshold + 1
    base_width, extra = divmod(64, segment_count)
    segments, offset = [], 0
    for index in range(segment_count):
        width = base_width + (1 if index < extra else 0)
        segments.append(((1 << width) - 1) << offset)
        offset += width

    def segment_buckets(records):
        result = [dict() for _ in segments]
        for item in records:
            for index, mask in enumerate(segments):
                result[index].setdefault(item["phash"] & mask, []).append(item)
        return result

    def segment_candidates(value, buckets):
        found = {}
        for index, mask in enumerate(segments):
            for item in buckets[index].get(value & mask, ()):
                found[item["image_id"]] = item
        return found.values()

    near_pairs, near_internal, ham_near_internal = [], [], []
    seen_cross, seen_isic, seen_ham = set(), set(), set()
    ham_buckets = segment_buckets([r for r in ham_records if r.get("phash") is not None])
    isic_buckets = segment_buckets(isic_records)
    for rec in isic_records:
        value = rec["phash"]
        for old in segment_candidates(value, ham_buckets):
            pair_key = (rec["image_id"], old["image_id"])
            if pair_key in seen_cross:
                continue
            seen_cross.add(pair_key)
            distance = (value ^ old["phash"]).bit_count()
            if distance <= near_threshold:
                near_pairs.append({"isic_image": rec["image_id"], "ham_image": old["image_id"],
                                   "ham_split": old["split"], "phash_hamming": distance})
        for old in segment_candidates(value, isic_buckets):
            if rec["image_id"] >= old["image_id"]:
                continue
            pair_key = (rec["image_id"], old["image_id"])
            if pair_key in seen_isic:
                continue
            seen_isic.add(pair_key)
            distance = (value ^ old["phash"]).bit_count()
            if distance <= near_threshold:
                near_internal.append({"image_a": rec["image_id"], "image_b": old["image_id"],
                                      "phash_hamming": distance})

    ham_buckets = segment_buckets([r for r in ham_records if r.get("phash") is not None])
    for rec in ham_records:
        if rec.get("phash") is None:
            continue
        for old in segment_candidates(rec["phash"], ham_buckets):
            if rec["image_id"] >= old["image_id"] or rec["split"] == old["split"]:
                continue
            pair_key = (rec["image_id"], old["image_id"])
            if pair_key in seen_ham:
                continue
            seen_ham.add(pair_key)
            distance = (rec["phash"] ^ old["phash"]).bit_count()
            if distance <= near_threshold:
                ham_near_internal.append({"image_a": rec["image_id"], "split_a": rec["split"],
                                          "image_b": old["image_id"], "split_b": old["split"],
                                          "phash_hamming": distance})

    source_counts = labels.source_dx.value_counts(dropna=False).to_dict()
    mapped_counts = labels[labels.dx.notna()].dx.value_counts().reindex(CLASSES, fill_value=0).to_dict()
    ham_class_counts = ham_metadata.dx.value_counts().reindex(CLASSES, fill_value=0).to_dict()
    patients = [c for c in metadata.columns if re.search(r"patient|subject", c, re.I)]
    lesion_col = next((c for c in metadata.columns if c.lower() in ("lesion_id", "lesionid")), None)
    grouped = pd.Series(False, index=labels.index)
    for col in patients + ([lesion_col] if lesion_col else []):
        grouped |= labels[col].fillna("").astype(str).str.strip().ne("")
    eligible_labels = (labels.label_status == "compatible") & labels.dx.notna() & labels.image_valid
    ham_image_errors = [r for r in ham_records if r.get("error")]
    image_content_fingerprint = hashlib.sha256("\n".join(
        f"{image_id}:{hashes[image_id]}" for image_id in sorted(hashes)
    ).encode("utf-8")).hexdigest()
    ham_content_fingerprint = hashlib.sha256("\n".join(
        f"{rec['image_id']}:{rec['split']}:{rec['sha256']}" for rec in sorted(ham_records, key=lambda r: (r["image_id"], r["split"]))
        if rec.get("sha256")
    ).encode("utf-8")).hexdigest()
    report = {
        "dataset": "ISIC 2019 Training",
        "dataset_version": "ISIC 2019 Training; identified from source CSV names and diagnosis columns",
        "dataset_root": str(root),
        "version_evidence": [gt_path.name, meta_path.name, "ISIC 2019 diagnosis columns"],
        "ground_truth_file": str(gt_path), "metadata_file": str(meta_path),
        "ground_truth_sha256": _sha256_file(gt_path),
        "metadata_sha256": _sha256_file(meta_path),
        "saved_split_manifest_sha256": _sha256_file(split_path),
        "metadata_columns": list(metadata.columns),
        "diagnosis_columns": source_diagnosis_columns,
        "unmapped_diagnosis_columns_excluded": unmapped_source_columns,
        "ground_truth_rows": int(len(labels)), "metadata_rows": int(len(metadata)),
        "image_files_found": int(len(image_map)), "valid_images": int(len(valid)),
        "invalid_images": invalid, "invalid_image_count": len(invalid), "metadata_without_image": missing_images,
        "invalid_ham_split_images": ham_image_errors,
        "ground_truth_rows_without_metadata": gt_without_metadata,
        "metadata_rows_without_ground_truth": metadata_without_gt,
        "ground_truth_rows_without_image": missing_images,
        "files_without_metadata": sorted(set(image_map) - set(metadata.image.astype(str))),
        "duplicate_image_filenames": duplicate_names,
        "source_diagnosis_counts": {str(k): int(v) for k, v in source_counts.items()},
        "unlabeled_rows": int(source_counts.get("unlabeled", 0)),
        "mapped_ham_class_counts": {str(k): int(v) for k, v in mapped_counts.items()},
        "excluded_incompatible_or_unlabeled": int((labels.label_status != "compatible").sum()),
        "excluded_by_source_diagnosis": {str(k): int(v) for k, v in source_counts.items()
                                          if ISIC_TO_HAM.get(str(k)) is None},
        "invalid_one_hot_rows": int((labels.label_status == "invalid_one_hot").sum()),
        "label_mapping": ISIC_TO_HAM,
        "patient_id_columns": patients,
        "lesion_id_column": lesion_col,
        "lesion_id_nonempty": int(metadata[lesion_col].notna().sum()) if lesion_col else 0,
        "lesion_id_unique": int(metadata[lesion_col].nunique()) if lesion_col else 0,
        "compatible_images_missing_patient_and_lesion_ids": int((eligible_labels & ~grouped).sum()),
        "ham_image_files_found": int(len(ham_map)), "ham_files_with_duplicate_names": ham_dupes,
        "ham_metadata_file": str(ham_metadata_path),
        "ham_metadata_rows": int(len(ham_metadata)),
        "ham_class_counts": {str(k): int(v) for k, v in ham_class_counts.items()},
        "ham_unlabeled_metadata_rows": int(ham_metadata.dx.isna().sum()),
        "shared_image_ids_with_ham": shared_image_ids,
        "shared_image_id_count_with_ham": len(shared_image_ids),
        "shared_image_ids_touching_test": int(sum(i in test_ids for i in shared_image_ids)),
        "preserved_test_images": int(len(test_ids)),
        "isic_image_content_fingerprint": image_content_fingerprint,
        "ham_split_content_fingerprint": ham_content_fingerprint,
        "exact_duplicate_pairs_with_ham": exact_pairs,
        "exact_duplicate_pairs_within_isic": exact_internal,
        "exact_duplicate_pairs_within_ham_across_splits": ham_exact_internal,
        "near_duplicate_pairs_with_ham": near_pairs,
        "near_duplicate_pairs_within_isic": near_internal,
        "near_duplicate_pairs_within_ham_across_splits": ham_near_internal,
        "near_duplicate_hamming_threshold": int(near_threshold),
        "test_overlap_exact": sum(p["ham_split"] == "test" for p in exact_pairs),
        "test_overlap_near": sum(p["ham_split"] == "test" for p in near_pairs),
        "ham_internal_test_overlap_exact": sum(p["split_a"] == "test" or p["split_b"] == "test" for p in ham_exact_internal),
        "ham_internal_test_overlap_near": sum(p["split_a"] == "test" or p["split_b"] == "test" for p in ham_near_internal),
        "integrity_verified": not invalid and not missing_images and not gt_without_metadata and not metadata_without_gt
                              and not duplicate_names and not ham_dupes and not ham_image_errors and not exact_pairs
                              and not near_pairs and not shared_image_ids and not exact_internal and not near_internal
                              and not ham_exact_internal and not ham_near_internal,
        "note": "Near-duplicate candidates use perceptual hash and require review. No patient ID is present in ISIC metadata.",
    }
    report["scan_signature"] = hashlib.sha256(
        json.dumps(_canonical(report), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return labels, ham_records, report


def load_isic_training_frame(labels):
    """Return only valid, explicitly compatible labeled rows for training."""
    keep = labels[(labels.label_status == "compatible") & labels.dx.notna() & labels.path.notna()
                  & labels.image_valid].copy()
    keep["y"] = keep.dx.map(HAM_TO_IDX).astype("int32")
    patient_col = next((c for c in keep.columns if re.search(r"patient|subject", c, re.I)), None)
    lesion_col = next((c for c in keep.columns if c.lower() in ("lesion_id", "lesionid")), None)
    patient_ids = keep[patient_col].astype("string").str.strip().replace("", pd.NA) if patient_col else pd.Series(pd.NA, index=keep.index)
    lesion_ids = keep[lesion_col].astype("string").str.strip().replace("", pd.NA) if lesion_col else pd.Series(pd.NA, index=keep.index)
    has_group = patient_ids.notna() | lesion_ids.notna()
    keep = keep[has_group].copy()
    patient_ids = patient_ids.loc[keep.index]
    lesion_ids = lesion_ids.loc[keep.index]
    keep["lesion_group"] = [f"patient:{p}" if pd.notna(p) else
                            (f"lesion:{l}" if pd.notna(l) else f"image:{i}")
                            for p, l, i in zip(patient_ids, lesion_ids, keep.image.astype(str))]
    return keep


def main():
    import argparse
    from src.dataset import load_config

    cfg = load_config()
    parser = argparse.ArgumentParser(description="Inspect ISIC metadata, images, labels and HAM overlap.")
    parser.add_argument("--isic-root", default=cfg.get("isic_dataset_root") or None)
    parser.add_argument("--ham-root", default=cfg["dataset_root"])
    parser.add_argument("--splits", default=cfg.get("existing_splits_path", "outputs/splits.csv"))
    parser.add_argument("--report", default=cfg.get("inspection_report_path", "outputs/isic_dataset_report.json"))
    parser.add_argument("--near-threshold", type=int, default=cfg.get("near_duplicate_hamming_threshold", 4))
    args = parser.parse_args()
    if not args.isic_root:
        parser.error("Set isic_dataset_root in config.json or provide --isic-root")
    _, _, report = inspect_isic(args.isic_root, args.ham_root, args.splits, _resolve(args.report), args.near_threshold)
    print(json.dumps({k: v for k, v in report.items() if k not in ("invalid_images", "exact_duplicate_pairs_with_ham",
                                                                    "exact_duplicate_pairs_within_isic",
                                                                    "exact_duplicate_pairs_within_ham_across_splits",
                                                                    "near_duplicate_pairs_with_ham",
                                                                    "near_duplicate_pairs_within_isic",
                                                                    "near_duplicate_pairs_within_ham_across_splits")}, indent=2))


if __name__ == "__main__":
    main()
