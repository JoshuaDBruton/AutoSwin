import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchxrayvision as xrv


class AdaptivePoolingLayer(nn.Module):
    def __init__(self, output_sizes=[1, 2, 4]):
        super().__init__()
        self.output_sizes = output_sizes

    def forward(self, x):
        batch_size, channels = x.size(0), x.size(1)
        pooled_features = []

        for size in self.output_sizes:
            pooled = F.adaptive_avg_pool2d(x, (size, size))
            pooled_features.append(pooled.view(batch_size, channels * size * size))

        return torch.cat(pooled_features, dim=1)


class TrxvWrapper(nn.Module):
    # Torch X-ray Vision Wrapper compatible with our MCCV
    # Tested with resnet50-res512-all
    def __init__(self, num_classes, model_name):
        super().__init__()

        xrvmodel = xrv.models.ResNet(weights=model_name)
        xrvmodel = xrvmodel.model
        in_features = xrvmodel.fc.in_features
        self.model = nn.Sequential(
                xrvmodel.conv1,
                xrvmodel.bn1,
                xrvmodel.layer1,
                xrvmodel.layer2,
                xrvmodel.layer3,
                xrvmodel.layer4,
        )

        self.spp = AdaptivePoolingLayer(output_sizes=[1, 2, 4])

        spp_output_size = in_features * (1*1 + 2*2 + 4*4)

        self.bn_after_spp = nn.LayerNorm(spp_output_size)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(spp_output_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        self.swin_with_adapter = None

    def initialize_weights(self, init_backbone=False):
        print("Randomly initialising weights")

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                module.weight.data *= 0.1   # Scaling down weights
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        if init_backbone:
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    module.weight.data *= 0.1  # reduce variance
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

    # text_features is just ignored because this has to match the FMs
    def forward(self, images, text_features=None, return_image_features=False):
        assert not torch.isnan(images).any(), "NaN in input images"
        assert not torch.isinf(images).any(), "Inf in input images"

        features = self.model(images)

        assert not torch.isnan(features).any(), "NaN in features"
        assert not torch.isinf(features).any(), "Inf in features"

        pooled_features = self.spp(features)
        pooled_features = self.bn_after_spp(pooled_features)

        assert not torch.isnan(pooled_features).any(), "NaN in pooled_features"
        assert not torch.isinf(pooled_features).any(), "Inf in pooled_features"

        logits = self.classifier(pooled_features)

        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"

        if return_image_features:
            return logits, pooled_features

        return logits

    def encode_text(self, text_tokens, normalize=False):
        return None

    def freeze_weights(self):
        return

    def print_parameter_count(self):
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        print("Parameter counts:")
        print(f"RESNET50 Backbone: {count_parameters(self.model):,}")
        print(f"Classifier: {count_parameters(self.classifier):,}")


def init_torchxrayvision_resnet_model(num_classes):
    model = TrxvWrapper(num_classes=num_classes, model_name='resnet50-res512-all')

    model.initialize_weights(init_backbone=True)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x + 1) / 2),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    return model, preprocess
