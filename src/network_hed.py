import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights


import torch
import torch.nn as nn
import torch.nn.functional as F

class SideOutput(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) # 1 output channel as binary mask

    def forward(self, x, target_size):
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class HED(nn.Module):
    def __init__(self, backbone, out_channels=1):
        super().__init__()
        self.backbone = backbone

        # Select which layers of the backbone to tap
        self.side1 = SideOutput(in_channels=64,out_channels=out_channels)   # example
        self.side2 = SideOutput(in_channels=128,out_channels=out_channels)
        self.side3 = SideOutput(in_channels=256,out_channels=out_channels)
        self.side4 = SideOutput(in_channels=512,out_channels=out_channels)
        self.side5 = SideOutput(in_channels=512,out_channels=out_channels)

        self.fuse = nn.Conv2d(5, 1, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2:]

        # Forward through backbone
        c1, c2, c3, c4, c5 = self.backbone(x)

        s1 = self.side1(c1, (H, W))
        s2 = self.side2(c2, (H, W))
        s3 = self.side3(c3, (H, W))
        s4 = self.side4(c4, (H, W))
        s5 = self.side5(c5, (H, W))

        # Concatenate side outputs
        fused = self.fuse(torch.cat([s1, s2, s3, s4, s5], dim=1))

        return [s1, s2, s3, s4, s5, fused]
    
class SimpleCNNBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleCNNBackbone, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.stage1(x)   # [B, 64, H, W]
        out2 = self.stage2(out1)  # [B, 128, H/2, W/2]
        out3 = self.stage3(out2)  # [B, 256, H/4, W/4]
        out4 = self.stage4(out3)  # [B, 512, H/8, W/8]
        out5 = self.stage5(out4)  # [B, 512, H/16, W/16]
        return [out1, out2, out3, out4, out5]




class HED_EfficientNet(nn.Module):
    def __init__(self, pretrained=True, custom_ckpt_path=None):
        super(HED_EfficientNet, self).__init__()

        # Load EfficientNet-B7
        if pretrained:
            backbone = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        else:
            backbone = efficientnet_b7(weights=None)

        if custom_ckpt_path:
            state_dict = torch.load(custom_ckpt_path, map_location='cpu')
            backbone.load_state_dict(state_dict)

        self.backbone = backbone.features  # Only the feature extractor part

        # Define the intermediate layers for deep supervision
        self.side_layers = nn.ModuleList([
            nn.Conv2d(32, 1, kernel_size=1),    # Stage 1
            nn.Conv2d(48, 1, kernel_size=1),    # Stage 2
            nn.Conv2d(80, 1, kernel_size=1),    # Stage 3
            nn.Conv2d(224, 1, kernel_size=1),   # Stage 4
            nn.Conv2d(640, 1, kernel_size=1),   # Stage 5
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_fuse = nn.Conv2d(5, 1, kernel_size=1)  # Fuse all side outputs

    def forward(self, x):
        side_outputs = []
        h = x

        # Stage 1
        h = self.backbone[0](h)
        side_outputs.append(self.side_layers[0](h))

        # Stage 2
        h = self.backbone[1](h)
        side_outputs.append(self.side_layers[1](h))

        # Stage 3
        h = self.backbone[2](h)
        h = self.backbone[3](h)
        side_outputs.append(self.side_layers[2](h))

        # Stage 4
        h = self.backbone[4](h)
        h = self.backbone[5](h)
        side_outputs.append(self.side_layers[3](h))

        # Stage 5
        h = self.backbone[6](h)
        h = self.backbone[7](h)
        side_outputs.append(self.side_layers[4](h))

        # Upsample all to input size
        side_outputs = [self.upsample(self.upsample(out)) for out in side_outputs]

        # Concatenate side outputs and fuse
        fuse = self.final_fuse(torch.cat(side_outputs, dim=1))

        return fuse, side_outputs
