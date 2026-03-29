import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as torchvision_models


class TVImageEncoder(nn.Module):
    """
    Load pretrained models via Huggingface, and extract features for different modalities.
    """
    def __init__(self, img_model: str = "resnet50"):
        super().__init__()
        print("=> Image extractor:")
        print("\t==> Image model: '{}'".format(img_model))

        def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
            if os.path.isfile(pretrained_weights):
                state_dict = torch.load(pretrained_weights, map_location="cpu")
                if checkpoint_key is not None and checkpoint_key in state_dict:
                    print(f"Take key {checkpoint_key} in provided checkpoint dict")
                    state_dict = state_dict[checkpoint_key]
                # remove `module.` prefix
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                # remove `backbone.` prefix induced by multicrop wrapper
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                msg = model.load_state_dict(state_dict, strict=False)
                print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
            else:
                print(
                    "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
                url = None
                if model_name == "vit_small" and patch_size == 16:
                    url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
                elif model_name == "vit_small" and patch_size == 8:
                    url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
                elif model_name == "vit_base" and patch_size == 16:
                    url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
                elif model_name == "vit_base" and patch_size == 8:
                    url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
                elif model_name == "xcit_small_12_p16":
                    url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
                elif model_name == "xcit_small_12_p8":
                    url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
                elif model_name == "xcit_medium_24_p16":
                    url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
                elif model_name == "xcit_medium_24_p8":
                    url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
                elif model_name == "resnet50":
                    url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
                if url is not None:
                    print(
                        "Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
                    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
                    model.load_state_dict(state_dict, strict=True)
                else:
                    print("There is no reference weights available for this model => We use random weights.")

        self.config = {}

        # DINO
        dino_model = torchvision_models.__dict__[img_model](num_classes=0)
        dino_model.fc = nn.Identity()
        dino_model.cuda()
        load_pretrained_weights(dino_model, "dino/dino_resnet50_pretrain.pth", "teacher", "resnet50", 16)
        self.img_model = dino_model.eval()

        self.dim = 2048
        # Freeze
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, pixel_values, normalize=True):
        self.eval()
        image_features = self.img_model(pixel_values)
        return F.normalize(image_features, dim=1) if normalize else image_features
