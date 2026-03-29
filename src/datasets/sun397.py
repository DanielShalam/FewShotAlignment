import os
import pickle

from .base_dataset import DATASET_REGISTRY
from .base_dataset import Datum, DatasetBase
from src.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class SUN397(DatasetBase):

    dataset_dir = "sun397"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg["root"]))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "1split_zhou_SUN397.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            classnames = []
            with open(os.path.join(self.dataset_dir, "ClassName.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()[1:].split("/")[-1].split(" ")[0]  # remove /
                    classnames.append(line)
            cname2lab = {c: i for i, c in enumerate(classnames)}
            print(cname2lab)
            trainval = self.read_data(cname2lab, "Training_01.txt", split_type="train")
            test = self.read_data(cname2lab, "Testing_01.txt", split_type="val")
            train, val = OxfordPets.split_trainval(trainval)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg["shots"]
        if num_shots >= 1:
            seed = cfg["seed"]
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)



        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, cname2lab, text_file, split_type: str):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                imname = "/".join(line.strip()[1:].split("/")[1:])  # remove /
                classname = os.path.dirname(imname)
                if "/" in classname:
                    imname = imname.replace(classname, classname.replace("/", "-"))

                label = cname2lab[classname.replace("/", "-")]
                impath = os.path.join(self.image_dir, split_type, imname)

                names = classname.split("/")
                names = names[::-1]  # put words like indoor/outdoor at first
                classname = " ".join(names)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
