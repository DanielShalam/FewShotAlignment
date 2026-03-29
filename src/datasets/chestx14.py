import glob
import csv
import math
import os
from pathlib import Path
from typing import List, Dict, Tuple
import random
from collections import Counter, defaultdict

from .base_dataset import Datum, DatasetBase, DATASET_REGISTRY
from src.utils import mkdir_if_missing, read_json, write_json


DROP_NO_FINDING = True

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
NO_FINDING = "No Finding"

def _read_csv_labels(csv_path: Path) -> Dict[str, List[str]]:
    """Returns dict: image_name -> list of labels (strings)."""
    out = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Image Index"]
            labs = [x.strip() for x in row["Finding Labels"].split("|")]
            out[name] = labs
    return out

def _read_list(list_path: Path) -> List[str]:
    with open(list_path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def _choose_primary_label(
    labels: List[str],
    global_freq: Dict[str, int],
    strategy: str = "rare",
    ignore_no_finding_in_mixed: bool = True,
) -> str:
    """Pick a single label from a possibly multi-label image."""
    labs = [l for l in labels if l in CLASS2IDX]
    if ignore_no_finding_in_mixed and len(labs) > 1 and "No Finding" in labs:
        labs = [l for l in labs if l != "No Finding"]
    if len(labs) > 1:
        return "No Finding"
    if not labs:
        # fallback: if nothing matched, treat as No Finding
        return "No Finding"

    if strategy == "first":
        return labs[0]
    elif strategy == "random":
        return random.choice(labs)
    elif strategy == "common":
        # pick the most frequent (could bias common pathologies)
        return max(labs, key=lambda l: global_freq.get(l, 0))
    else:  # "rare" (default): pick the rarest among this image's labels
        return min(labs, key=lambda l: global_freq.get(l, 0))


def _build_split(
    names: List[str],
    name2labels: Dict[str, List[str]],
    resolve_impath_fn,               # <--- new
    global_freq: Dict[str, int],
    primary_strategy: str,
    ignore_no_finding: bool,
) -> List[Datum]:
    out = []
    num_dropped = 0
    for nm in names:
        p = resolve_impath_fn(nm)
        if p is None:
            # skip cleanly if missing
            continue
        labs = name2labels.get(nm, ["No Finding"])
        primary = _choose_primary_label(labs, global_freq, primary_strategy, ignore_no_finding)
        if DROP_NO_FINDING and primary == NO_FINDING:
            num_dropped += 1
            continue
        out.append(Datum(impath=p, label=CLASS2IDX[primary], classname=primary.lower()))

    print("Length of split: {} ({} dropped)".format(len(out), num_dropped))
    return out


@DATASET_REGISTRY.register()
class ChestX14(DatasetBase):
    """
    NIH ChestX-ray14, single-label conversion for Dassl pipeline.

    Expected structure at ROOT:
      ROOT/
        images/                      # all images here (as provided by NIH)
        Data_Entry_2017.csv
        train_val_list.txt
        test_list.txt

    Config (optional):
      cfg.DATASET.CHESTX:
        PRIMARY_STRATEGY: "rare" | "first" | "random" | "common"
        VAL_SIZE_PER_CLASS: 200      # #val images per class drawn from train_val
        SEED: 1
        IGNORE_NO_FINDING_IN_MIXED: True
    """

    dataset_dir = "ChestX14"  # will be ROOT/dataset_dir unless ROOT points directly

    def __init__(self, cfg):
        self.root = Path(cfg["root"])
        self.name = "ChestX14"

        # resolve files
        # allow ROOT to be the true folder (with CSV and images) or to contain ChestX14/
        base = self.root
        if not (base / "Data_Entry_2017.csv").exists():
            base = self.root / self.dataset_dir

        csv_path = base / "Data_Entry_2017.csv"
        train_val_list = base / "train_val_list.txt"
        test_list = base / "test_list.txt"

        # collect possible image roots (supports both official 'images/' and Kaggle 'images_0xx/')
        img_roots = []
        if (base / "images").exists():
            img_roots.append(base / "images")
        img_roots.extend(sorted(Path(p) for p in glob.glob(str(base / "images_*")) if Path(p).is_dir()))
        if not img_roots:
            raise FileNotFoundError(f"No image folder found under {base} (tried 'images/' and 'images_*').")

        # fast-ish resolver with cache so we don’t scan every time
        _path_cache = {}

        def resolve_impath(name: str) -> str | None:
            if name in _path_cache:
                return _path_cache[name]
            for root in img_roots:
                p = root / "images" / name
                if p.exists():
                    _path_cache[name] = str(p)
                    return _path_cache[name]
            return None

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expect files at {base}: Data_Entry_2017.csv (got {csv_path.exists()})"
            )
        if not train_val_list.exists() or not test_list.exists():
            raise FileNotFoundError(
                f"Expect official split lists at {base}: train_val_list.txt and test_list.txt."
            )

        # parse
        name2labels = _read_csv_labels(csv_path)
        trainval_names = _read_list(train_val_list)
        test_names = _read_list(test_list)

        # compute global label frequency on train_val for 'rare/common' strategies
        freq = Counter()
        for nm in trainval_names:
            for l in name2labels.get(nm, ["No Finding"]):
                if l in CLASS2IDX:
                    freq[l] += 1

        # options
        opt = getattr(cfg.DATASET, "CHESTX", None)
        primary_strategy = (getattr(opt, "PRIMARY_STRATEGY", "rare") if opt is not None else "rare")
        ignore_no_finding = (getattr(opt, "IGNORE_NO_FINDING_IN_MIXED", True) if opt is not None else True)
        val_per_class = int(getattr(opt, "VAL_SIZE_PER_CLASS", 4) if opt is not None else 4)

        # filter names by assigning primary labels
        primaries = {}
        per_class = defaultdict(list)
        trainval_names_filtered = []
        for nm in trainval_names:
            labs = name2labels.get(nm, ["No Finding"])
            primary = _choose_primary_label(labs, freq, primary_strategy, ignore_no_finding)
            if DROP_NO_FINDING and primary == "No Finding":
                continue
            primaries[nm] = primary
            per_class[primary].append(nm)
            trainval_names_filtered.append(nm)

        test_names_filtered = []
        for nm in test_names:
            labs = name2labels.get(nm, ["No Finding"])
            primary = _choose_primary_label(labs, freq, primary_strategy, ignore_no_finding)
            if DROP_NO_FINDING and primary == NO_FINDING:
                continue
            test_names_filtered.append(nm)

        print(per_class.keys())
        print([f"{c}: {len(items)}" for c, items in per_class.items()])

        # build Datum lists
        print("Build train and val...")
        train = _build_split(trainval_names_filtered, name2labels, resolve_impath, freq, primary_strategy, ignore_no_finding)
        train, val = self.split_trainval(train)
        print("Build test...")
        test = _build_split(test_names_filtered, name2labels, resolve_impath, freq, primary_strategy, ignore_no_finding)

        # train&val to few-shot datasets
        k = cfg["shots"]
        if k >= 1:
            train = self.generate_fewshot_dataset(train, num_shots=k)
            # val = self.generate_fewshot_dataset(val, num_shots=min(k, val_per_class))
            data = {"train": train, "val": val}

        # # 2) draw per-class validation subset
        # val_names, tr_names = [], []
        # random.seed(rng_seed)
        # for c, items in per_class.items():
        #     random.shuffle(items)
        #     train_slice = items[:num_shots]
        #     val_slice = items[num_shots: num_shots + min(num_shots, val_per_class)]
        #     tr_names.extend(train_slice)
        #     val_names.extend(val_slice)

        # # 3) build Datum lists
        # print("Build train_x...")
        # train_x = _build_split(tr_names, name2labels, resolve_impath, freq, primary_strategy, ignore_no_finding)
        # print("Build val...")
        # val = _build_split(val_names, name2labels, resolve_impath, freq, primary_strategy, ignore_no_finding)
        # print("Build test...")
        # test = _build_split(test_names, name2labels, resolve_impath, freq, primary_strategy, ignore_no_finding)

        # metadata for prompts

        self._classnames = CLASSES if not DROP_NO_FINDING else [c for c in CLASSES if c != "No Finding"]

        train, val, test = self.subsample_classes(train, val, test, subsample='all')

        print(f"[ChestX14] Found {len(img_roots)} image roots: {[r.name for r in img_roots][:3]} ...")
        super().__init__(train_x=train, val=val, test=test)

    @property
    def classnames(self) -> List[str]:
        return self._classnames

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output
