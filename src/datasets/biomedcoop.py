"""BiomedCoOp single-label image-classification datasets.

Each instance in cfg['data_root']/<NAME> has:
  - class subdirs with images
  - split_<NAME>.json from the BiomedCoOp repo with {'train','val','test': [[relpath, lbl, classname],...]}
"""
import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import random

from .base_dataset import DATASET_REGISTRY, Datum, DatasetBase


def _load(root: Path):
    sj = next(root.glob("split_*.json"))
    s = json.loads(sj.read_text())
    class_map = {}
    for rel, lbl, cn in s["train"]:
        class_map[lbl] = cn
    classnames = [class_map[i] for i in sorted(class_map)]
    def mk(items):
        return [Datum(impath=str(root / rel), label=int(lbl), classname=cn) for rel, lbl, cn in items]
    return classnames, mk(s["train"]), mk(s.get("val", [])), mk(s["test"])


class _BiomedCoOpBase(DatasetBase):
    _subdir = ""  # override
    def __init__(self, cfg):
        root = Path(cfg["root"]) / self._subdir
        classnames, all_train, val, test = _load(root)
        self._classnames = classnames

        # few-shot K per class, seeded
        K = int(cfg.get("shots", 16))
        rng = random.Random(int(cfg.get("seed", 1)))
        per_class: Dict[int, List[Datum]] = defaultdict(list)
        for d in all_train:
            per_class[d.label].append(d)
        train_x = []
        for c in range(len(classnames)):
            items = per_class.get(c, [])[:]
            rng.shuffle(items)
            train_x.extend(items[:K])

        super().__init__(train_x=train_x, val=val, test=test)

    @property
    def classnames(self) -> List[str]:
        return self._classnames


@DATASET_REGISTRY.register()
class BUSI(_BiomedCoOpBase):
    _subdir = "BUSI"

@DATASET_REGISTRY.register()
class KneeXray(_BiomedCoOpBase):
    _subdir = "KneeXray"

@DATASET_REGISTRY.register()
class CHMNIST(_BiomedCoOpBase):
    _subdir = "CHMNIST"

@DATASET_REGISTRY.register()
class BTMRI(_BiomedCoOpBase):
    _subdir = "BTMRI"

@DATASET_REGISTRY.register()
class COVID_19(_BiomedCoOpBase):
    _subdir = "COVID_19"

@DATASET_REGISTRY.register()
class CTKidney(_BiomedCoOpBase):
    _subdir = "CTKidney"

@DATASET_REGISTRY.register()
class DermaMNIST(_BiomedCoOpBase):
    _subdir = "DermaMNIST"

@DATASET_REGISTRY.register()
class Kvasir(_BiomedCoOpBase):
    _subdir = "Kvasir"

@DATASET_REGISTRY.register()
class LungColon(_BiomedCoOpBase):
    _subdir = "LungColon"

@DATASET_REGISTRY.register()
class OCTMNIST(_BiomedCoOpBase):
    _subdir = "OCTMNIST"

@DATASET_REGISTRY.register()
class RETINA(_BiomedCoOpBase):
    _subdir = "RETINA"