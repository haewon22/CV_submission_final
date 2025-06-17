import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, bias=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, d, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn = nn.BatchNorm2d(in_ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.act(self.bn(x))
        x = self.pw(x)
        return x

class MultiDilationSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=2, bias=True):
        super().__init__()
        p1 = k // 2
        p2 = p1 + (d - 1) * (k - 1) // 2
        self.dw1 = nn.Conv2d(in_ch, in_ch, k, 1, p1, groups=in_ch, bias=False)
        self.dw2 = nn.Conv2d(in_ch, in_ch, k, 1, p2, d, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_ch, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(in_ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x1 = self.act(self.bn1(self.dw1(x)))
        x2 = self.act(self.bn2(self.dw2(x)))
        return self.pw(x1 + x2)

class SEModule(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        attn = self.net(self.pool(x))
        return x * attn

class MicroDownsampleModuleV2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.use_pool = in_ch < out_ch
        conv_out = out_ch if not self.use_pool else out_ch - in_ch
        self.conv = SeparableConv2d(in_ch, conv_out, k=3, s=2, p=1)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        y = self.conv(x)
        if self.use_pool:
            y = torch.cat([y, F.max_pool2d(x, 2, 2)], dim=1)
        return self.act(self.bn(y))

class MicroResidualConvModule(nn.Module):
    def __init__(self, ch, dil=1, drop=0.):
        super().__init__()
        self.conv = SeparableConv2d(ch, ch, 3, 1, dil, dil, False)
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3)
        self.drop = nn.Dropout2d(drop)
        self.act = nn.SiLU(inplace=True)
        self.se = SEModule(ch)
    def forward(self, x):
        y = self.drop(self.bn(self.conv(x)))
        y = self.se(y)
        return self.act(x + y)

class MicroResidualMultiDilationConvModule(nn.Module):
    def __init__(self, ch, dil=2, drop=0.):
        super().__init__()
        self.conv = MultiDilationSeparableConv2d(ch, ch, 3, dil, False)
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3)
        self.drop = nn.Dropout2d(drop)
        self.act = nn.SiLU(inplace=True)
        self.se = SEModule(ch)
    def forward(self, x):
        y = self.drop(self.bn(self.conv(x)))
        y = self.se(y)
        return self.act(x + y)

class MicroUpsampleModuleV2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = SeparableConv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        return self.act(self.bn(x))

class GradientFeatureModule(nn.Module):
    def __init__(self, ch):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(ch, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=torch.float32).expand(ch, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.refine = SeparableConv2d(ch * 2, ch, 1, 1, 0, 1, False)
        self.bn     = nn.BatchNorm2d(ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=x.size(1))
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=x.size(1))
        g  = torch.cat([gx, gy], dim=1)
        return self.act(self.bn(self.refine(g)))

class MicroNetV5EncoderV2(nn.Module):
    def __init__(self, in_ch: int, ch: tuple = (10, 20, 26), rates: tuple = (1, 2, 4, 8)):
        super().__init__()
        c1, c2, c3 = ch
        self.down1 = MicroDownsampleModuleV2(in_ch, c1)
        self.grad  = GradientFeatureModule(c1)
        self.down2 = MicroDownsampleModuleV2(c1, c2)
        self.mid   = nn.Sequential(MicroResidualConvModule(c2, 1, 0.0), MicroResidualConvModule(c2, 1, 0.0))
        self.down3 = MicroDownsampleModuleV2(c2, c3)
        self.ctx   = nn.Sequential(*[MicroResidualMultiDilationConvModule(c3, d, 0.1) for d in rates])
    def forward(self, x):
        d1 = self.down1(x)
        d1 = d1 + self.grad(d1)
        d2 = self.mid(self.down2(d1))
        d3 = self.down3(d2)
        out = self.ctx(d3)
        return out, d2

class submission_20221377(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 ch: tuple = (6, 10, 18),
                 interpolate: bool = True):
        super().__init__()
        self.interpolate = interpolate
        c1, c2, c3 = ch

        self.encoder = MicroNetV5EncoderV2(in_channels, ch=ch)
        self.aux_ds  = MicroDownsampleModuleV2(in_channels, c1)
        self.aux_ref = MicroResidualConvModule(c1, 1, 0.0)

        self.up1     = MicroUpsampleModuleV2(c3, c2)
        self.up_mid  = nn.Sequential(
            MicroResidualConvModule(c2, 1, 0.0),
            MicroResidualConvModule(c2, 1, 0.0)
        )

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head_conv = SeparableConv2d(c1 + c2, num_classes, k=3, s=1, p=1)

    def forward(self, x):
        input_size = x.shape[2:]
        enc, skip = self.encoder(x)
        aux = self.aux_ref(self.aux_ds(x))
        y = self.up1(enc)
        if y.size(2) != skip.size(2) or y.size(3) != skip.size(3):
            y = F.interpolate(y, size=skip.shape[2:], mode='bilinear', align_corners=False)
        y = y + skip
        y = self.up_mid(y)
        y = self.final_up(y)
        y = torch.cat([y, aux], dim=1)
        out = self.head_conv(y)
        if self.interpolate and out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)
        return out

if __name__ == "__main__":
    ch_cfg = (6, 12, 18)
    img_size = 256
    net_fixed = submission_MiniNetV9(in_channels=3, num_classes=21, ch=ch_cfg)
    x = torch.randn(1, 3, img_size, img_size)
    y = net_fixed(x)
    params = sum(p.numel() for p in net_fixed.parameters() if p.requires_grad)
    print(f"출력 크기 : {y.shape}")
    print(f"파라미터 : {params/1e3:.2f} K")
