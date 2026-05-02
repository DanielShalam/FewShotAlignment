"""DeepLoc 2.0 (multi-label) dataset using precomputed ESM + text caches.

Expects under cfg['root']:
  labels_multihot.pt, seq_embed_<esm_tag>.pt, class_text_<text_tag>.pt, meta.json
Config keys:
  root:            dir with the above files
  esm_file:        filename inside root for protein embeddings
  text_file:       filename inside root for class-text prototypes (relative, used by model cfg['text_features_path'])
  shots:           K for few-shot support (per primary class)
  val_per_class:   (optional, default 4)
  seed:            (optional, default 1)
"""
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from collections import defaultdict

from .base_dataset import DATASET_REGISTRY, FEATURE_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class DeepLoc2(DatasetBase):
    def __init__(self, cfg):
        self.cfg = cfg
        root = Path(cfg["root"])
        meta = json.loads((root / "meta.json").read_text())
        self._classnames: List[str] = list(meta["classes"])
        C = len(self._classnames)

        seq = torch.load(root / cfg["esm_file"], weights_only=False)
        lab = torch.load(root / "labels_multihot.pt", weights_only=False)

        self.features: Dict[str, torch.Tensor] = {
            k: F.normalize(seq[k].float(), dim=-1) for k in ("train", "validation", "test")
        }
        self.labels_mh: Dict[str, torch.Tensor] = {k: lab[k].float() for k in ("train", "validation", "test")}

        self.multi_map: Dict[str, torch.Tensor] = {}
        def _impath(split: str, idx: int) -> str:
            return f"feat://{split}/{idx}"

        def _register(split: str, i: int) -> str:
            ip = _impath(split, i)
            FEATURE_REGISTRY[ip] = self.features[split][i]
            self.multi_map[ip] = self.labels_mh[split][i]
            return ip

        # test: all
        test = []
        for i in range(self.features["test"].shape[0]):
            ip = _register("test", i)
            y = self.labels_mh["test"][i]
            primary = int(torch.argmax(y).item())
            test.append(Datum(impath=ip, label=primary, classname=self._classnames[primary]))

        # few-shot K per primary class from the train split
        shots = int(cfg.get("shots", 16))
        rng = random.Random(int(cfg.get("seed", 1)))
        Y_tr = self.labels_mh["train"]
        per_class: Dict[int, List[int]] = defaultdict(list)
        for i in range(Y_tr.shape[0]):
            per_class[int(torch.argmax(Y_tr[i]).item())].append(i)
        tr_idx = []
        for c in range(C):
            items = per_class.get(c, []).copy()
            rng.shuffle(items)
            tr_idx.extend(items[:shots])
        train_x = []
        for i in tr_idx:
            ip = _register("train", i)
            primary = int(torch.argmax(Y_tr[i]).item())
            train_x.append(Datum(impath=ip, label=primary, classname=self._classnames[primary]))

        # val: use the official DeepLoc validation split in full
        val = []
        Y_val = self.labels_mh["validation"]
        for i in range(self.features["validation"].shape[0]):
            ip = _register("validation", i)
            primary = int(torch.argmax(Y_val[i]).item())
            val.append(Datum(impath=ip, label=primary, classname=self._classnames[primary]))

        self.dim_img = int(self.features["train"].shape[1])
        super().__init__(train_x=train_x, val=val, test=test)

    @property
    def classnames(self) -> List[str]:
        return self._classnames
