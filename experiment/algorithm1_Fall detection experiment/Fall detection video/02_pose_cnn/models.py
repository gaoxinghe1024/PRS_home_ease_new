"""P1 SmallUNet, P2 CPM3Stage and P3 SimpleBaseline definitions."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_H, OUT_W = (64, 48)
NUM_KPTS = 17


class ConvBNReLU(nn.Sequential):

    def __init__(self, cin, cout, k=3, s=1, groups=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


def _init_from_scratch(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class SmallUNet(nn.Module):

    def __init__(self, c_in=3, width=(32, 64, 128, 256)):
        super().__init__()
        w1, w2, w3, w4 = width
        self.enc1 = nn.Sequential(ConvBNReLU(c_in, w1), ConvBNReLU(w1, w1))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(ConvBNReLU(w1, w2), ConvBNReLU(w2, w2))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(ConvBNReLU(w2, w3), ConvBNReLU(w3, w3))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(ConvBNReLU(w3, w4), ConvBNReLU(w4, w4))
        self.pool4 = nn.MaxPool2d(2)
        self.mid = ConvBNReLU(w4, w4)
        self.up3 = nn.ConvTranspose2d(w4, w3, 2, 2)
        self.dec3 = nn.Sequential(ConvBNReLU(w3 + w4, w3), ConvBNReLU(w3, w3))
        self.up2 = nn.ConvTranspose2d(w3, w2, 2, 2)
        self.dec2 = nn.Sequential(ConvBNReLU(w2 + w3, w2), ConvBNReLU(w2, w2))
        self.head = nn.Conv2d(w2, NUM_KPTS, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        m = self.mid(self.pool4(e4))
        d3 = self.dec3(torch.cat([self.up3(m), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e3], dim=1))
        return [self.head(d2)]


class CPMStage(nn.Module):

    def __init__(self, cin, c=256):
        super().__init__()
        layers = [ConvBNReLU(cin, c, 1)] + [ConvBNReLU(c, c, 3) for _ in range(4)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(c, NUM_KPTS, 1)

    def forward(self, x):
        return self.head(self.body(x))


class CPM3Stage(nn.Module):

    def __init__(self, c_in=3, c_front=128, c_stage=160, num_stages=3):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv2d(c_in, c_front, 7, 4, 3, bias=False),
            nn.BatchNorm2d(c_front),
            nn.ReLU(inplace=True),
            ConvBNReLU(c_front, c_front),
        )
        ch = c_front
        self.stages = nn.ModuleList()
        for t in range(num_stages):
            self.stages.append(CPMStage(ch, c_stage))
            ch = c_front + NUM_KPTS

    def forward(self, x):
        f = self.front(x)
        hms, prev = ([], None)
        for t, st in enumerate(self.stages):
            inp = f if t == 0 else torch.cat([f, prev], dim=1)
            h = st(inp)
            hms.append(h)
            prev = h
        return hms


class BasicBlock(nn.Module):

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + idt, inplace=True)


class SlimResNet(nn.Module):

    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.layer1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))
        self.layer2 = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 64))
        self.layer3 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer4 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.out_ch = 256

    def forward(self, x):
        y = self.pool(self.stem(x))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        return y


class SimpleBaseline(nn.Module):

    def __init__(self, c_in=3, deconv_ch=256):
        super().__init__()
        self.backbone = SlimResNet(c_in)
        ch = self.backbone.out_ch
        deconvs = []
        for _ in range(3):
            deconvs += [
                nn.ConvTranspose2d(ch, deconv_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(deconv_ch),
                nn.ReLU(inplace=True),
            ]
            ch = deconv_ch
        self.deconv = nn.Sequential(*deconvs)
        self.head = nn.Conv2d(deconv_ch, NUM_KPTS, 1)

    def forward(self, x):
        return [self.head(self.deconv(self.backbone(x)))]


class FullResNet18(nn.Module):

    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.out_ch = 512

    def forward(self, x):
        y = self.pool(self.stem(x))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        return y


class SimpleBaselineFull(nn.Module):

    def __init__(self, c_in=3, deconv_ch=256):
        super().__init__()
        self.backbone = FullResNet18(c_in)
        deconvs = []
        ch = self.backbone.out_ch
        for _ in range(3):
            deconvs += [
                nn.ConvTranspose2d(ch, deconv_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(deconv_ch),
                nn.ReLU(inplace=True),
            ]
            ch = deconv_ch
        self.deconv = nn.Sequential(*deconvs)
        self.head = nn.Conv2d(deconv_ch, NUM_KPTS, 1)

    def forward(self, x):
        return [self.head(self.deconv(self.backbone(x)))]


class PretrainedResNet18Baseline(nn.Module):
    PRETRAINED_PATH = os.path.expanduser(
        "~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    )

    def __init__(self, deconv_ch=256):
        super().__init__()
        import torchvision.models as tv

        bb = tv.resnet18()
        if os.path.exists(self.PRETRAINED_PATH):
            bb.load_state_dict(torch.load(self.PRETRAINED_PATH, map_location="cpu"))
            self._has_pretrained = True
        else:
            self._has_pretrained = False
        self.backbone = nn.Sequential(
            bb.conv1,
            bb.bn1,
            bb.relu,
            bb.maxpool,
            bb.layer1,
            bb.layer2,
            bb.layer3,
            bb.layer4,
        )
        deconvs = []
        ch = 512
        for _ in range(3):
            deconvs += [
                nn.ConvTranspose2d(ch, deconv_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(deconv_ch),
                nn.ReLU(inplace=True),
            ]
            ch = deconv_ch
        self.deconv = nn.Sequential(*deconvs)
        self.head = nn.Conv2d(deconv_ch, NUM_KPTS, 1)
        self.deconv.apply(_init_from_scratch)
        self.head.apply(_init_from_scratch)

    def forward(self, x):
        return [self.head(self.deconv(self.backbone(x)))]


def build_model(name):
    name = name.lower()
    if name in ("p1", "unet", "smallunet"):
        m, n = (SmallUNet(), 1)
    elif name in ("p2", "cpm", "cpm3stage"):
        m, n = (CPM3Stage(), 3)
    elif name in ("p3", "sb", "simplebaseline"):
        m, n = (SimpleBaseline(), 1)
    elif name in ("p4", "full", "simplebaselinefull", "resnet18"):
        m, n = (SimpleBaselineFull(), 1)
    elif name in ("p5", "pretrained", "pretrainedresnet18"):
        return (PretrainedResNet18Baseline(), 1)
    else:
        raise ValueError(f"unknown architecture {name}")
    m.apply(_init_from_scratch)
    return (m, n)


def count_params(m):
    return sum((p.numel() for p in m.parameters() if p.requires_grad))


if __name__ == "__main__":
    for name in ("P1", "P2", "P3"):
        m, ns = build_model(name)
        x = torch.randn(2, 3, 256, 192)
        outs = m(x)
        shapes = [tuple(o.shape) for o in outs]
        print(
            f"{name}: params={count_params(m) / 1000000.0:.2f}M stages={len(outs)} out={shapes}"
        )
