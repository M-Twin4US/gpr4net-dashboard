"""
D1-B1 Model — HyperbolaV24Lite adapted for 3-channel D1-A3 input.

Input : (B, 3, 256, 256)
          Ch0: norm εr (EPS_MAX=10)   ─┐
          Ch1: binary water mask       ─┴─ Path 1 (2-ch encoder)
          Ch2: norm σ                  ──── Path 2 (1-ch encoder, minor adaptation)

Output: (B, 1, 256, 256)  B-scan

Architecture: identical to HyperbolaV24Lite except enc2's first block takes
1 input channel instead of 2 (difference: 126 fewer parameters, negligible).

The channel split mirrors the original model's design philosophy:
  Original: Path 1 = [εr, sat_mask],  Path 2 = [σ, zeros(PEC)]
  D1-B1:    Path 1 = [εr, water_mask], Path 2 = [σ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj  = nn.Conv2d(embed_dim, embed_dim, 1)
        self.k_proj  = nn.Conv2d(embed_dim, embed_dim, 1)
        self.v_proj  = nn.Conv2d(embed_dim, embed_dim, 1)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.norm    = nn.LayerNorm(embed_dim)

    def forward(self, query, kv):
        B, C, H, W = query.shape
        Q = self.q_proj(query).reshape(B, self.num_heads, self.head_dim, H * W)
        K = self.k_proj(kv).reshape(B, self.num_heads, self.head_dim, H * W)
        V = self.v_proj(kv).reshape(B, self.num_heads, self.head_dim, H * W)

        attn = F.softmax(torch.matmul(Q.transpose(-2, -1), K) * self.scale, dim=-1)
        out  = torch.matmul(attn, V.transpose(-2, -1))
        out  = out.transpose(-2, -1).reshape(B, C, H, W)
        out  = self.out_proj(out)

        res = query.permute(0, 2, 3, 1)
        out = out.permute(0, 2, 3, 1)
        return self.norm(res + out).permute(0, 3, 1, 2)


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ca1 = MultiHeadCrossAttention(embed_dim, num_heads)
        self.ca2 = MultiHeadCrossAttention(embed_dim, num_heads)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 1)

    def forward(self, f1, f2):
        return self.fusion(torch.cat([self.ca1(f1, f2), self.ca2(f2, f1)], dim=1))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


class D1B1Model(nn.Module):
    """
    HyperbolaV24Lite adapted for 3-channel D1-A3 input.

    Path 1 (2-ch): [εr, water_mask]  — spatial contrast + saturation state
    Path 2 (1-ch): [σ]               — conductivity stream

    Cross-attention fuses both streams at the bottleneck.
    Decoder uses Path 1 skip connections.
    """

    def __init__(self, out_channels=1, features=[14, 28, 56, 112, 224], num_heads=4):
        super().__init__()

        # Path 1 encoder — 2 input channels (εr + water_mask)
        self.enc1_1 = EncoderBlock(2,           features[0])
        self.enc1_2 = EncoderBlock(features[0], features[1])
        self.enc1_3 = EncoderBlock(features[1], features[2])
        self.enc1_4 = EncoderBlock(features[2], features[3])
        self.enc1_5 = EncoderBlock(features[3], features[4])

        # Path 2 encoder — 1 input channel (σ)
        self.enc2_1 = EncoderBlock(1,           features[0])
        self.enc2_2 = EncoderBlock(features[0], features[1])
        self.enc2_3 = EncoderBlock(features[1], features[2])
        self.enc2_4 = EncoderBlock(features[2], features[3])
        self.enc2_5 = EncoderBlock(features[3], features[4])

        # Cross-attention fusion at bottleneck
        self.fusion = AdaptiveFeatureFusion(features[4], num_heads)

        # Decoder (skip connections from Path 1)
        self.dec1 = DecoderBlock(features[4], features[3])
        self.dec2 = DecoderBlock(features[3], features[2])
        self.dec3 = DecoderBlock(features[2], features[1])
        self.dec4 = DecoderBlock(features[1], features[0])

        self.outc = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        # Split channels
        p1 = x[:, 0:2]   # [εr, water_mask]
        p2 = x[:, 2:3]   # [σ]

        # Path 1 (saves skips for decoder)
        s1, x1 = self.enc1_1(p1)
        s2, x1 = self.enc1_2(x1)
        s3, x1 = self.enc1_3(x1)
        s4, x1 = self.enc1_4(x1)
        _,  f1 = self.enc1_5(x1)

        # Path 2
        _, x2 = self.enc2_1(p2)
        _, x2 = self.enc2_2(x2)
        _, x2 = self.enc2_3(x2)
        _, x2 = self.enc2_4(x2)
        _, f2 = self.enc2_5(x2)

        # Fuse at bottleneck
        x = self.fusion(f1, f2)

        # Decode with Path 1 skips
        x = self.dec1(x, s4)
        x = self.dec2(x, s3)
        x = self.dec3(x, s2)
        x = self.dec4(x, s1)

        x = F.interpolate(self.outc(x), size=(256, 256),
                          mode='bilinear', align_corners=True)
        return torch.sigmoid(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = D1B1Model(features=[14, 28, 56, 112, 224], num_heads=4)
    n = count_parameters(model)
    print(f'D1-B1 (HyperbolaV24Lite 3-ch): {n:,} parameters ({n/1e6:.2f}M)')
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    print(f'Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}')
