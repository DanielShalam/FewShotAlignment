import os
import csv
import glob
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

import numpy as np
import torch
from PIL import Image


from .base_dataset import DATASET_REGISTRY
from .base_dataset import Datum, DatasetBase


# ----------------------------
# Utilities
# ----------------------------

def _read_csv_labels(csv_path: Path, agg: str = "ANY") -> Dict[str, List[str]]:
    """
    Read VinDr 'wide' multi-hot CSV (multiple rows per image_id, one per rad_id)
    and aggregate to a single multi-label list per image.

    Supported agg:
      - "ANY": a label is positive if ANY radiologist marked 1
      - "MAJORITY": positive if (sum >= ceil(n_raters / 2))
    Returns: dict: image_id -> List[str] of positive label names (canonicalized case)
    """
    import math
    out: Dict[str, List[str]] = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in reader.fieldnames]

        # VinDr schema check
        if "image_id" not in cols:
            raise ValueError(f"{csv_path} missing 'image_id' column.")
        # pathology columns = everything except these meta fields
        meta_cols = {"image_id", "rad_id"}
        patho_cols = [c for c in cols if c not in meta_cols]

        # accumulate per image_id
        sums: Dict[str, Dict[str, int]] = {}
        counts: Dict[str, int] = {}

        for row in reader:
            img = row["image_id"].strip()
            if img == "":
                continue
            if img not in sums:
                sums[img] = {p: 0 for p in patho_cols}
                counts[img] = 0
            counts[img] += 1
            for p in patho_cols:
                try:
                    v = int(row[p])
                except Exception:
                    try:
                        v = 1 if float(row[p]) > 0 else 0
                    except Exception:
                        v = 0
                sums[img][p] += v

        # decide positives per image_id
        for img, vec in sums.items():
            nrat = max(1, counts[img])
            thresh = 1 if agg.upper() == "ANY" else math.ceil(nrat / 2.0)
            pos = []
            for p, s in vec.items():
                if s >= thresh:
                    pos.append(p.strip())
            # canonicalize names (lower noise)
            pos = _normalize_labels(pos)
            if len(pos) == 0:
                pos = ["No finding"]
            out[img] = pos

    return out


def _dicom_to_png_cached(dicom_path: Path, cache_png_root: Path) -> str:
    """
    Convert DICOM to PNG under cache folder preserving relative layout.
    Returns the PNG path as str. If already converted, returns cached file.
    """

    # construct destination path
    rel = dicom_path.name.replace(".dicom", ".png")
    dst = cache_png_root / rel
    if dst.exists():
        return str(dst)

    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_modality_lut
        from pydicom.pixel_data_handlers import apply_windowing
    except Exception as e:
        raise ImportError("pydicom is required for VinDr-CXR. `pip install pydicom`") from e

    dst.parent.mkdir(parents=True, exist_ok=True)

    dcm = pydicom.filereader.dcmread(str(dicom_path))
    arr = apply_modality_lut(dcm.pixel_array, dcm)
    try:
        arr = apply_windowing(arr, dcm)
    except Exception:
        # Some files may miss window tags; fall back to raw
        pass

    # Photometric interpretation: MONOCHROME1 means inverse
    photo = dcm.get((0x28, 0x04), None)
    mode = photo.value if photo is not None else "MONOCHROME2"

    arr = np.asarray(arr)
    arr = arr.astype(np.float32)
    # normalize 0..255 for PNG saving (robust percentile scaling)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    if mode == "MONOCHROME1":
        arr = 1.0 - arr  # invert

    img = (arr * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="L")  # grayscale
    pil.save(str(dst))
    return str(dst)


# ----------------------------
# Label spaces
# ----------------------------

# Full VinDr label set (image-level) commonly used.
VINDR_LABELS = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Lesion",
    "Pleural effusion",
    "Effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "No finding",
]

# The "clean 6" categories often used for classification papers / rad-DINO:
CLEAN6 = [
    "Lung Opacity",
    "Cardiomegaly",
    "Pleural Thickening",   # normalized canonical name
    "Aortic Enlargement",   # normalized canonical name
    "Pulmonary Fibrosis",
    "Tuberculosis",
    "Pleural Effusion"
]

# Map raw labels/synonyms -> canonical (full space and clean-6 space)
NORM_MAP = {
    # pleura
    "Pleural_Thickening": "Pleural thickening",
    "Pleural Thickening": "Pleural thickening",
    "Pleural effusion": "Pleural effusion",
    "Effusion": "Pleural effusion",
    # aortic
    "Aortic enlargement": "Aortic enlargement",
    "Aortic Enlargement": "Aortic enlargement",
    # lung opacity
    "Opacity": "Lung Opacity",
    # fibrosis
    "Pulmonary fibrosis": "Pulmonary fibrosis",
    # TB (only present in some VinDr derivatives)
    "Tuberculosis": "Tuberculosis",
    # keep exact matches too
}

# map canonical->clean6 canonical for subset switch
TO_CLEAN6 = {
    "Lung Opacity": "Lung Opacity",
    "Cardiomegaly": "Cardiomegaly",
    "Pleural thickening": "Pleural Thickening",
    "Aortic enlargement": "Aortic Enlargement",
    "Pulmonary fibrosis": "Pulmonary Fibrosis",
    "Tuberculosis": "Tuberculosis",
    "Pleural effusion": "Pleural Effusion",
}

def _normalize_labels(raw_labels: List[str]) -> List[str]:
    out = []
    for l in raw_labels:
        l0 = NORM_MAP.get(l, l).strip()
        # print(l, l0)
        out.append(l0)
    return out


# ----------------------------
# Few-shot helpers
# ----------------------------

def _choose_primary_label(
    labels: List[str],
    allowed: List[str],
    global_freq: Dict[str, int],
    strategy: str = "rare",
    drop_no_finding: bool = True,
) -> str | None:
    """Pick a single primary label from possibly multi-label list."""
    labs = [l for l in labels if l in allowed]
    if drop_no_finding:
        labs = [l for l in labs if l.lower() != "no finding"]
    if not labs:
        return None
    if strategy == "first":
        return labs[0]
    if strategy == "random":
        return random.choice(labs)
    if strategy == "common":
        return max(labs, key=lambda l: global_freq.get(l, 0))
    # default: "rare"
    return min(labs, key=lambda l: global_freq.get(l, 0))


# ----------------------------
# Dataset
# ----------------------------

@DATASET_REGISTRY.register()
class VinDrCXR(DatasetBase):
    """
    VinDr-CXR → Dassl few-shot classification (single-label view).
    Folder layout:
      ROOT/
        train/*.dicom
        test/*.dicom
        annotations/
          image_labels_train.csv
          image_labels_test.csv
          annotations_train.csv
          annotations_test.csv

    Config options (add under cfg.VINDR):
      USE_CLEAN6: bool = False          # if True, reduce to 6 categories
      DROP_NO_FINDING: bool = True
      PRIMARY_STRATEGY: str = "rare"    # rare | first | random | common
      VAL_SIZE_PER_CLASS: int = 4
      DICOM_CACHE_SUBDIR: str = "_png_cache"   # where to store converted PNGs
    """

    dataset_dir = "VinDrCXR"

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg["root"])
        base = self.root
        if not (base / "annotations").exists():
            # allow ROOT/dataset_dir fallback
            base = self.root / self.dataset_dir

        anndir = base / "annotations"
        train_dir = base / "train"
        test_dir = base / "test"

        # sanity
        for p in [anndir, train_dir, test_dir]:
            if not p.exists():
                raise FileNotFoundError(f"[VinDrCXR] Missing path: {p}")

        # Options
        opt = getattr(cfg.DATASET, "VINDR", None)
        use_clean6 = bool(getattr(opt, "USE_CLEAN6", True)) if opt else True
        drop_no_finding = bool(getattr(opt, "DROP_NO_FINDING", True)) if opt else True
        primary_strategy = (getattr(opt, "PRIMARY_STRATEGY", "rare") if opt else "rare")
        val_per_class = int(getattr(opt, "VAL_SIZE_PER_CLASS", 4) if opt else 4)
        cache_subdir = getattr(opt, "DICOM_CACHE_SUBDIR", "_png_cache") if opt else "_png_cache"
        agg_policy = (getattr(opt, "AGG", "ANY") if opt else "ANY")  # "ANY" or "MAJORITY"

        # label space
        if use_clean6:
            classnames = CLEAN6.copy()
            allowed = set(TO_CLEAN6.values())
        else:
            classnames = [c for c in VINDR_LABELS if (drop_no_finding is False or c != "No finding")]
            allowed = set(classnames)

        self._classnames = classnames
        CLASS2IDX = {c: i for i, c in enumerate(classnames)}

        # Read CSVs
        tr_csv = anndir / "image_labels_train.csv"
        te_csv = anndir / "image_labels_test.csv"
        if not tr_csv.exists() or not te_csv.exists():
            raise FileNotFoundError("[VinDrCXR] Expected image_labels_{train,test}.csv in annotations/")

        name2labels_tr = _read_csv_labels(tr_csv, agg=agg_policy)
        name2labels_te = _read_csv_labels(te_csv, agg=agg_policy)

        # # Normalize labels
        # def norm_map(d: Dict[str, List[str]]) -> Dict[str, List[str]]:
        #     out = {}
        #     for k, v in d.items():
        #         out[k] = _normalize_labels(v)
        #     return out
        # name2labels_tr = norm_map(name2labels_tr)
        # name2labels_te = norm_map(name2labels_te)

        # If using clean-6, collapse normalized labels to the clean6 set
        if use_clean6:
            def collapse_to_clean6(v: List[str]) -> List[str]:
                out = []
                for l in v:
                    if l in TO_CLEAN6:
                        out.append(TO_CLEAN6[l])
                return list(sorted(set(out))) or (["No finding"] if not drop_no_finding else [])
            name2labels_tr = {k: collapse_to_clean6(v) for k, v in name2labels_tr.items()}
            name2labels_te = {k: collapse_to_clean6(v) for k, v in name2labels_te.items()}

        # Build index of actual dicom files
        def _index_split(split_dir: Path) -> Dict[str, Path]:
            exts = ["*.dicom"]
            files = {}
            n_miss = 0
            for e in exts:
                for p in glob.glob(str(split_dir / e)):
                    pth = Path(p)
                    stem = pth.stem
                    if stem.endswith(".dicom"):  # handle ".dicom.gz" → stem ".dicom"
                        stem = Path(stem).stem
                    if Path(os.path.join(split_dir, stem+".dicom")).exists():
                        files[stem] = pth
                    else:
                        n_miss += 1

            print(split_dir, f". {len(list(files.keys()))} files loaded, {n_miss} files missing")
            return files

        train_files = _index_split(train_dir)
        test_files = _index_split(test_dir)

        # Compute frequency on train labels for primary strategy
        freq = Counter()
        for img_id, labs in name2labels_tr.items():
            for l in labs:
                if l in allowed:
                    freq[l] += 1

        # Few-shot split (deterministic)
        rng = random.Random(int(getattr(cfg, "SEED", 1)))

        # Group train images by primary label
        per_class = defaultdict(list)
        for img_id, labs in name2labels_tr.items():
            if img_id not in train_files:
                continue
            primary = _choose_primary_label(
                labs, list(allowed), freq,
                strategy=primary_strategy,
                drop_no_finding=drop_no_finding
            )
            if primary is None:
                continue
            per_class[primary].append(img_id)

        # Draw shots + validation per class
        shots = int(getattr(cfg.DATASET, "NUM_SHOTS", 1))
        tr_names, val_names = [], []
        for c in classnames:
            items = per_class.get(c, [])
            rng.shuffle(items)
            tr_slice  = items[:shots]
            val_slice = items[shots: shots + min(shots, val_per_class)]
            tr_names.extend(tr_slice)
            val_names.extend(val_slice)

        # Build cached PNG paths
        cache_train = base / cache_subdir / "train"
        cache_test  = base / cache_subdir / "test"
        cache_train.mkdir(parents=True, exist_ok=True)
        cache_test.mkdir(parents=True, exist_ok=True)

        def to_png_path(img_id: str, split: str) -> str | None:
            if split == "train":
                dicom_path = train_files.get(img_id)
                cache_root = cache_train
            else:
                dicom_path = test_files.get(img_id)
                cache_root = cache_test
            if dicom_path is None or not dicom_path.exists():
                return None
            try:
                return _dicom_to_png_cached(dicom_path, cache_root)
            except Exception:
                return None

        # Build Datum lists
        def build(names: List[str], name2labels: Dict[str, List[str]]) -> List[Datum]:
            out: List[Datum] = []
            miss = 0
            for nm in names:
                imp = to_png_path(nm, "train")
                if imp is None:
                    miss += 1
                    continue
                # primary label (again) for datum
                labs = name2labels.get(nm, [])
                primary = _choose_primary_label(
                    labs, list(allowed), freq,
                    strategy=primary_strategy,
                    drop_no_finding=drop_no_finding
                )
                if primary is None:
                    continue
                out.append(Datum(impath=imp, label=CLASS2IDX[primary], classname=primary))
            if miss > 0:
                print(f"[VinDrCXR] Warning: {miss} train files failed to convert/read (skipped).")
            return out

        def build_test_all(name2labels: Dict[str, List[str]]) -> List[Datum]:
            out: List[Datum] = []
            miss = 0
            for nm in sorted(name2labels.keys()):
                imp = to_png_path(nm, "test")
                if imp is None:
                    miss += 1
                    continue
                labs = name2labels.get(nm, [])
                primary = _choose_primary_label(
                    labs, list(allowed), freq,
                    strategy=primary_strategy,
                    drop_no_finding=drop_no_finding
                )
                if primary is None:
                    continue
                out.append(Datum(impath=imp, label=CLASS2IDX[primary], classname=primary))
            if miss > 0:
                print(f"[VinDrCXR] Warning: {miss} test files failed to convert/read (skipped).")
            return out

        print("[VinDrCXR] Building train_x ...")
        train_x = build(tr_names, name2labels_tr)

        print("[VinDrCXR] Building val ...")
        val = build(val_names, name2labels_tr)

        print("[VinDrCXR] Building test ...")
        test = build_test_all(name2labels_te)

        self._six = CLEAN6
        self._six2idx = {c: i for i, c in enumerate(self._six)}

        # image_id -> list[str] positives (after your ANY/MAJORITY aggregation)
        self.name2labels_train = name2labels_tr
        self.name2labels_test = name2labels_te

        def labels_to_multi(labs: list[str]) -> torch.Tensor:
            y = torch.zeros(len(self._six), dtype=torch.float32)
            for L in labs:
                if L in self._six2idx:
                    y[self._six2idx[L]] = 1.0
            return y

        def path_to_id(p: str) -> str:
            # your resolver used file stem as image_id; keep consistent
            return Path(p).stem

        self.multi_map = {}  # impath -> tensor[6]
        for d in (train_x + val + test):
            img_id = path_to_id(d.impath)
            labs = self.name2labels_train.get(img_id, self.name2labels_test.get(img_id, []))
            self.multi_map[d.impath] = labels_to_multi(labs)

        # optional: class frequencies on train_x for pos_weight
        cnt = torch.zeros(len(CLEAN6))
        for d in train_x:
            cnt += self.multi_map[d.impath]
        self.train_pos_counts = cnt  # tensor[6]
        self.train_num = max(1, len(train_x))

        # Summary
        def per_class_counts(data: List[Datum]) -> Dict[str, int]:
            cnt = Counter([d.classname for d in data])
            return {k: cnt.get(k, 0) for k in classnames}

        print(f"[VinDrCXR] classnames: {classnames}")
        print(f"[VinDrCXR] train_x: {len(train_x)} | per-class: {per_class_counts(train_x)}")
        print(f"[VinDrCXR] val:     {len(val)}     | per-class: {per_class_counts(val)}")
        print(f"[VinDrCXR] test:    {len(test)}    | per-class: {per_class_counts(test)}")

        super().__init__(train_x=train_x, val=val, test=test)

    @property
    def classnames(self) -> List[str]:
        return self._classnames
