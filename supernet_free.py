import copy
import os

try:
    import torch
    import torch.nn as nn
    from torchprofile import profile_macs
except Exception:
    torch = None
    nn = None
    profile_macs = None


if nn is not None:
    class MBConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, expansion, stride):
            super().__init__()
            hidden = int(in_channels * expansion)
            self.use_residual = stride == 1 and in_channels == out_channels
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            )
            self.depthwise = nn.Sequential(
                nn.Conv2d(
                    hidden,
                    hidden,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=hidden,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            )
            self.project = nn.Sequential(
                nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            out = self.project(self.depthwise(self.expand(x)))
            if self.use_residual:
                out = out + x
            return out


    class SurrogateNet(nn.Module):
        def __init__(self, arch, n_classes=1000):
            super().__init__()
            stage_channels = [24, 40, 80, 112, 160]
            stage_strides = [1, 2, 2, 1, 2]
            self.stem = nn.Sequential(
                nn.Conv2d(3, stage_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stage_channels[0]),
                nn.ReLU(inplace=True),
            )
            blocks = []
            in_channels = stage_channels[0]
            idx = 0
            for stage_idx, depth in enumerate(arch["d"]):
                out_channels = stage_channels[stage_idx]
                for block_idx in range(depth):
                    stride = stage_strides[stage_idx] if block_idx == 0 else 1
                    blocks.append(
                        MBConvBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=int(arch["ks"][idx]),
                            expansion=float(arch["e"][idx]),
                            stride=stride,
                        )
                    )
                    in_channels = out_channels
                    idx += 1
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, 960, kernel_size=1, bias=False),
                nn.BatchNorm2d(960),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(960, n_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            x = self.head(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)
else:
    class SurrogateNet:
        def __init__(self, arch, n_classes=1000):
            raise ImportError("torch is required to build SurrogateNet")


def architecture_to_model_config(arch, n_classes=1000):
    stage_channels = [24, 40, 80, 112, 160]
    stage_strides = [1, 2, 2, 1, 2]
    blocks = []
    idx = 0
    for stage_idx, depth in enumerate(arch["d"]):
        for block_idx in range(depth):
            blocks.append(
                {
                    "stage": stage_idx,
                    "kernel_size": int(arch["ks"][idx]),
                    "expansion": float(arch["e"][idx]),
                    "stride": stage_strides[stage_idx] if block_idx == 0 else 1,
                    "out_channels": stage_channels[stage_idx],
                }
            )
            idx += 1
    return {
        "n_classes": int(n_classes),
        "resolution": int(arch["r"]),
        "stage_channels": stage_channels,
        "blocks": blocks,
    }


class ComplexityProfiler:
    def __init__(self, n_classes=1000):
        self.n_classes = n_classes
        self._cache = {}

    @staticmethod
    def _key(arch):
        return (
            tuple(arch["ks"]),
            tuple(arch["e"]),
            tuple(arch["d"]),
            int(arch["r"]),
        )

    def profile(self, arch, sec_obj):
        key = (self._key(arch), sec_obj)
        if key in self._cache:
            return self._cache[key]
        use_proxy =os.environ.get("NSGANET_FAST_PROFILE", "0").strip().lower() in ("1", "true", "yes")
        if torch is None or profile_macs is None or use_proxy:
            proxy = float(sum(arch["ks"]) * sum(arch["e"]) * sum(arch["d"]) * (int(arch["r"]) ** 2))
            flops = proxy / 1e7
            params = proxy / 5e8
        else:
            model = SurrogateNet(arch, n_classes=self.n_classes).eval()
            with torch.no_grad():
                inputs = torch.randn(1, 3, int(arch["r"]), int(arch["r"]))
                flops = float(profile_macs(copy.deepcopy(model), (inputs,)) / 1e6)
                params = float(sum(parameter.numel() for parameter in model.parameters()) / 1e6)
        if sec_obj == "params":
            value = params
        elif sec_obj in ("cpu", "gpu"):
            value = flops
        else:
            value = flops
        self._cache[key] = value
        return value
