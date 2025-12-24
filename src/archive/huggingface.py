# Save model files in huggingface format

import os
import argparse
import torch
from transformers import PretrainedConfig, PreTrainedModel
import network_hed as hed  


# -----------------------------
# Config wrapper
# -----------------------------
class HEDConfig(PretrainedConfig):
    model_type = "hed"

    def __init__(self, in_channels=4, backbone="ResNet50",
                 freeze_backbone=False, guidance=False, loss_function="BCE", date=None, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.guidance = guidance
        self.loss_function = loss_function
        self.date = date


# -----------------------------
# Model wrapper
# -----------------------------
class HEDModel(PreTrainedModel):
    config_class = HEDConfig

    def __init__(self, config):
        super().__init__(config)

        if config.backbone == "SimpleCNN":
            backbone = hed.SimpleCNNBackbone(in_channels=config.in_channels)
        else:
            backbone = hed.ResNet50Backbone(
                in_channels=config.in_channels,
                backbone_dataset=config.backbone
            )

        self.hed = hed.HED(
            backbone=backbone,
            in_channels=config.in_channels,
            out_channels=1
        )

    def forward(self, x):
        return self.hed(x)


# -----------------------------
# Main logic
# -----------------------------
def wrap_and_save(model_path, save_folder):
    model_name = os.path.basename(model_path)
    name_split = model_name.split('_')

    # Parse filename like your get_model
    date = name_split[1]
    model_type = name_split[2]
    backbone_type = name_split[3]
    freeze_backbone = name_split[4]

    guidance = name_split[5]
    if guidance == "guided":
        guidance = True
        in_channels = 5
    else:
        guidance = False
        in_channels = 4

    loss_function = name_split[6].split('.')[0]

    # Build config
    config = HEDConfig(
        in_channels=in_channels,
        backbone=backbone_type,
        freeze_backbone=freeze_backbone,
        guidance=guidance,
        loss_function=loss_function,
        date=date,
    )

      # ---- Load the raw HED model ----
    if backbone_type == "SimpleCNN":
        backbone = hed.SimpleCNNBackbone(in_channels=in_channels)
    else:
        backbone = hed.ResNet50Backbone(
            in_channels=in_channels,
            backbone_dataset=backbone_type
        )
    raw_model = hed.HED(backbone=backbone, in_channels=in_channels, out_channels=1)

    state_dict = torch.load(model_path, map_location="cpu")
    raw_model.load_state_dict(state_dict)

    # ---- Wrap in Hugging Face PreTrainedModel ----
    hf_model = HEDModel(config)
    hf_model.hed = raw_model  # plug in the loaded weights

    # Save in Hugging Face format
    os.makedirs(save_folder, exist_ok=True)
    hf_model.save_pretrained(save_folder)
    config.save_pretrained(save_folder)

    print(f"✅ Model and config saved to {save_folder}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .pth model file")
    parser.add_argument("--save_folder", type=str, required=True,
                        help="Where to save Hugging Face model")
    args = parser.parse_args()

    wrap_and_save(args.model_path, args.save_folder)
