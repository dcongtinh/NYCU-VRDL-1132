import torch
from collections import OrderedDict
import torch.nn as nn
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, FeaturePyramidNetwork
import timm


class CSPIntermediateLayer(nn.ModuleDict):
    def __init__(self, model: nn.Module) -> None:

        layers = OrderedDict()
        layers['stem'] = model.stem
        layers['0'] = model.stages[0]
        layers['1'] = model.stages[1]
        layers['2'] = model.stages[2]
        layers['3'] = model.stages[3]

        super(CSPIntermediateLayer, self).__init__(layers)

        self.return_layers = ['0', '1', '2', '3']

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out[name] = x
        return out


class CSPWithFPN(nn.Module):
    def __init__(self):
        super(CSPWithFPN, self).__init__()

        extra_blocks = LastLevelMaxPool()

        # backbone = timm.models.cspresnext50(pretrained=True)
        backbone = timm.models.cspresnet50(pretrained=True)

        layers_to_train = ['stages.1', 'stages.2', 'stages.3']
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        self.body = CSPIntermediateLayer(backbone)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512, 1024],
            out_channels=256,
            extra_blocks=extra_blocks,
        )
        self.out_channels = 256

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class EffIntermediateLayer(nn.ModuleDict):
    def __init__(self, model: nn.Module, n_layers) -> None:

        layers = OrderedDict()
        layers['conv_stem'] = model.conv_stem
        layers['bn1'] = model.bn1
        layers['0'] = model.blocks[0]
        layers['1'] = model.blocks[1]
        layers['2'] = model.blocks[2]
        layers['3'] = model.blocks[3]
        layers['4'] = model.blocks[4]
        layers['5'] = model.blocks[5]
        layers['conv_head'] = model.conv_head
        layers['bn2'] = model.bn2

        super(EffIntermediateLayer, self).__init__(layers)

        self.return_layers = [str(i) for i in range(6)]

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out[name] = x
        return out


class EfficientNetBackbone(torch.nn.Module):
    def __init__(self, variant, out_channels=256):
        super().__init__()
        # Pretrained EfficientNetV2-S feature extractor
        # efficientnetv2_rw_s.ra2_in1k
        # tf_efficientnetv2_m
        # tf_efficientnetv2_s
        # efficientnet_b2
        backbone = timm.create_model(variant, pretrained=True)
        # print([name for name, _ in backbone.named_parameters()])
        # print(aa)
        n_layers = len(backbone.blocks)
        layers_to_train = [f'blocks.{i}' for i in range(n_layers)]
        for name, parameter in backbone.named_parameters():
            print(name)
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        # in_channels from the 4 returned feature maps
        if 'efficientnetv2' in variant:
            # tf_efficientnetv2_s.in21k_ft_in1k
            self.in_channels_list = [24, 48, 64, 128]
        if variant == 'efficientnet_b2':
            # Typical for EfficientNet-B2, efficientnet_b2
            self.in_channels_list = [16, 24, 48, 120]

        self.body = EffIntermediateLayer(backbone, n_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool()
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        # print('body.x', x['3'].shape)

        x = self.fpn(x)
        # print('fpn.x', x['pool'].shape)
        # print(aa)
        return x
