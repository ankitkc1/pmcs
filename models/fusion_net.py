import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import general_conv3d, prm_generator, prm_generator_laststage, region_aware_modal_fusion

BASE_CHANNELS = 16
NUM_MODALITIES = 4


class Encoder(nn.Module):
    """Single-modality 3D conv encoder, four downsampling stages."""

    def __init__(self, channel=1):
        super(Encoder, self).__init__()

        self.e1_c1 = general_conv3d(channel, BASE_CHANNELS, pad_type='reflect')
        self.e1_c2 = general_conv3d(BASE_CHANNELS, BASE_CHANNELS, pad_type='reflect')
        self.e1_c3 = general_conv3d(BASE_CHANNELS, BASE_CHANNELS, pad_type='reflect')

        self.e2_c1 = general_conv3d(BASE_CHANNELS, BASE_CHANNELS*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d(BASE_CHANNELS*2, BASE_CHANNELS*2, pad_type='reflect')
        self.e2_c3 = general_conv3d(BASE_CHANNELS*2, BASE_CHANNELS*2, pad_type='reflect')

        self.e3_c1 = general_conv3d(BASE_CHANNELS*2, BASE_CHANNELS*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d(BASE_CHANNELS*4, BASE_CHANNELS*4, pad_type='reflect')
        self.e3_c3 = general_conv3d(BASE_CHANNELS*4, BASE_CHANNELS*4, pad_type='reflect')

        self.e4_c1 = general_conv3d(BASE_CHANNELS*4, BASE_CHANNELS*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d(BASE_CHANNELS*8, BASE_CHANNELS*8, pad_type='reflect')
        self.e4_c3 = general_conv3d(BASE_CHANNELS*8, BASE_CHANNELS*8, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        return x1, x2, x3, x4


class Decoder(nn.Module):
    """Plain single-stream decoder, used privately per modality (never shared/aggregated)."""

    def __init__(self, num_cls=4):
        super(Decoder, self).__init__()

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(BASE_CHANNELS*8, BASE_CHANNELS*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(BASE_CHANNELS*8, BASE_CHANNELS*4, pad_type='reflect')
        self.d3_out = general_conv3d(BASE_CHANNELS*4, BASE_CHANNELS*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(BASE_CHANNELS*4, BASE_CHANNELS*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(BASE_CHANNELS*4, BASE_CHANNELS*2, pad_type='reflect')
        self.d2_out = general_conv3d(BASE_CHANNELS*2, BASE_CHANNELS*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(BASE_CHANNELS*2, BASE_CHANNELS, pad_type='reflect')
        self.d1_c2 = general_conv3d(BASE_CHANNELS*2, BASE_CHANNELS, pad_type='reflect')
        self.d1_out = general_conv3d(BASE_CHANNELS, BASE_CHANNELS, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=BASE_CHANNELS, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4):
        de_x4 = self.d3_c1(self.d3(x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred


class FusionAdapter(nn.Module):
    """
    Personalisation head applied to the fully-aggregated fusion decoder's
    output feature map, conditioned on which modalities are available.

    Every learnable tensor is split into a `prior` half (uploaded to the
    server and FedAvg-aggregated with every other client's prior each round)
    and a `residual` half (trained locally and never uploaded). The tensor
    actually used at forward time is `prior + residual`, so a client always
    builds on the latest shared prior while keeping a private local
    correction that the server never sees.
    """
    def __init__(self, channels, num_modalities=NUM_MODALITIES, cond_dim=32):
        super(FusionAdapter, self).__init__()
        self.channels = channels

        def dual(*shape):
            return nn.Parameter(torch.zeros(*shape)), nn.Parameter(torch.zeros(*shape))

        self.conv_weight_prior, self.conv_weight_residual = dual(channels, channels, 3, 3, 3)
        nn.init.kaiming_normal_(self.conv_weight_prior)
        self.conv_bias_prior, self.conv_bias_residual = dual(channels)

        self.cond1_weight_prior, self.cond1_weight_residual = dual(cond_dim, num_modalities)
        nn.init.kaiming_normal_(self.cond1_weight_prior)
        self.cond1_bias_prior, self.cond1_bias_residual = dual(cond_dim)

        self.cond2_weight_prior, self.cond2_weight_residual = dual(channels * 2, cond_dim)
        self.cond2_bias_prior, self.cond2_bias_residual = dual(channels * 2)

    def prior_parameters(self):
        return [p for n, p in self.named_parameters() if n.endswith('_prior')]

    def residual_parameters(self):
        return [p for n, p in self.named_parameters() if n.endswith('_residual')]

    def forward(self, x, mask):
        conv_weight = self.conv_weight_prior + self.conv_weight_residual
        conv_bias = self.conv_bias_prior + self.conv_bias_residual
        out = F.conv3d(x, conv_weight, conv_bias, padding=1)

        cond1_weight = self.cond1_weight_prior + self.cond1_weight_residual
        cond1_bias = self.cond1_bias_prior + self.cond1_bias_residual
        cond2_weight = self.cond2_weight_prior + self.cond2_weight_residual
        cond2_bias = self.cond2_bias_prior + self.cond2_bias_residual

        h = F.relu(F.linear(mask.float(), cond1_weight, cond1_bias))
        gamma, beta = F.linear(h, cond2_weight, cond2_bias).chunk(2, dim=-1)
        gamma = gamma.view(-1, self.channels, 1, 1, 1)
        beta = beta.view(-1, self.channels, 1, 1, 1)

        out = out * (1 + gamma) + beta
        return F.relu(out, inplace=True) + x


class FusionDecoder(nn.Module):
    """Region-aware multi-modal fusion decoder, ending in the personalised FusionAdapter."""

    def __init__(self, num_cls=4):
        super(FusionDecoder, self).__init__()

        self.d3_c1 = general_conv3d(BASE_CHANNELS*8, BASE_CHANNELS*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(BASE_CHANNELS*8, BASE_CHANNELS*4, pad_type='reflect')
        self.d3_out = general_conv3d(BASE_CHANNELS*4, BASE_CHANNELS*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d(BASE_CHANNELS*4, BASE_CHANNELS*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(BASE_CHANNELS*4, BASE_CHANNELS*2, pad_type='reflect')
        self.d2_out = general_conv3d(BASE_CHANNELS*2, BASE_CHANNELS*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d(BASE_CHANNELS*2, BASE_CHANNELS, pad_type='reflect')
        self.d1_c2 = general_conv3d(BASE_CHANNELS*2, BASE_CHANNELS, pad_type='reflect')
        self.d1_out = general_conv3d(BASE_CHANNELS, BASE_CHANNELS, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=BASE_CHANNELS, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

        self.RFM4 = region_aware_modal_fusion(in_channel=BASE_CHANNELS*8, num_cls=num_cls)
        self.RFM3 = region_aware_modal_fusion(in_channel=BASE_CHANNELS*4, num_cls=num_cls)
        self.RFM2 = region_aware_modal_fusion(in_channel=BASE_CHANNELS*2, num_cls=num_cls)
        self.RFM1 = region_aware_modal_fusion(in_channel=BASE_CHANNELS*1, num_cls=num_cls)

        self.prm_generator4 = prm_generator_laststage(in_channel=BASE_CHANNELS*8, num_cls=num_cls)
        self.prm_generator3 = prm_generator(in_channel=BASE_CHANNELS*4, num_cls=num_cls)
        self.prm_generator2 = prm_generator(in_channel=BASE_CHANNELS*2, num_cls=num_cls)
        self.prm_generator1 = prm_generator(in_channel=BASE_CHANNELS*1, num_cls=num_cls)

        self.adapter = FusionAdapter(BASE_CHANNELS, num_modalities=NUM_MODALITIES)

    def forward(self, x1, x2, x3, x4, mask):
        # x1 - [B, 4, 16,  N,   H,   W]
        # x2 - [B, 4, 32,  N/2, H/2, W/2]
        # x3 - [B, 4, 64,  N/4, H/4, W/4]
        # x4 - [B, 4, 128, N/8, H/8, W/8]
        # mask - [B, 4] bool, which of the 4 modalities are available
        prm_pred4 = self.prm_generator4(x4, mask)
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask)
        fusion_x4 = de_x4

        de_x4 = self.d3_c1(self.up2(de_x4))

        if de_x4.shape[2:] != x3.shape[3:]:
            _, _, _, H, W, Z = x3.size()
            de_x4 = de_x4[:, :, :H, :W, :Z]

        prm_pred3 = self.prm_generator3(de_x4, x3, mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), mask)

        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        fusion_x3 = de_x3

        de_x3 = self.d2_c1(self.up2(de_x3))

        if de_x3.shape[2:] != x2.shape[3:]:
            _, _, _, H, W, Z = x2.size()
            de_x3 = de_x3[:, :, :H, :W, :Z]

        prm_pred2 = self.prm_generator2(de_x3, x2, mask)
        de_x2 = self.RFM2(x2, prm_pred2.detach(), mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        fusion_x2 = de_x2

        de_x2 = self.d1_c1(self.up2(de_x2))

        if de_x2.shape[2:] != x1.shape[3:]:
            _, _, _, H, W, Z = x1.size()
            de_x2 = de_x2[:, :, :H, :W, :Z]

        prm_pred1 = self.prm_generator1(de_x2, x1, mask)
        de_x1 = self.RFM1(x1, prm_pred1.detach(), mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))
        fusion_x1 = de_x1

        fused = self.adapter(de_x1, mask)     # personalised prior+residual adapter, conditioned on the modality mask

        logits = self.seg_layer(fused)
        pred = self.softmax(logits)

        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3),
                      self.up8(prm_pred4)), (fusion_x1, fusion_x2, fusion_x3, fusion_x4), fused


class FusionSegNet(nn.Module):
    """
    Four modality-specific encoders feeding a shared FusionDecoder.

    `encode()` and `decode()` are split out so a caller can run the encoder
    once per sample and the (mask-conditioned) decoder more than once, e.g.
    for the modality-dropout self-distillation used during local training.
    """
    def __init__(self, num_cls=4):
        super(FusionSegNet, self).__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.fusion_decoder = FusionDecoder(num_cls=num_cls)
        self.modality_decoder = Decoder(num_cls=num_cls)   # private per-modality decoder, never shared/aggregated

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    def encode(self, x):
        f_x1, f_x2, f_x3, f_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
        c_x1, c_x2, c_x3, c_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])

        x1 = torch.stack((f_x1, c_x1, t1_x1, t2_x1), dim=1)   # [B, 4, 16,  N,   H,   W]
        x2 = torch.stack((f_x2, c_x2, t1_x2, t2_x2), dim=1)
        x3 = torch.stack((f_x3, c_x3, t1_x3, t2_x3), dim=1)
        x4 = torch.stack((f_x4, c_x4, t1_x4, t2_x4), dim=1)

        per_modal = (
            (f_x1, f_x2, f_x3, f_x4),
            (c_x1, c_x2, c_x3, c_x4),
            (t1_x1, t1_x2, t1_x3, t1_x4),
            (t2_x1, t2_x2, t2_x3, t2_x4),
        )
        return x1, x2, x3, x4, per_modal

    def decode(self, x1, x2, x3, x4, mask):
        return self.fusion_decoder(x1, x2, x3, x4, mask)

    def forward(self, x, mask, fx1=None, fx2=None, fx3=None, fx4=None):
        # fx1..fx4 are accepted (and ignored) purely for call-site compatibility.
        x1, x2, x3, x4, per_modal = self.encode(x)

        fuse_pred, prm_preds, fusion_preds, fused_repr = self.decode(x1, x2, x3, x4, mask)

        if self.is_training:
            per_modal_preds = torch.stack([self.modality_decoder(*feats) for feats in per_modal], dim=0)
            msk_preds = per_modal_preds[mask[0], ...]
            return fuse_pred, prm_preds, fusion_preds, msk_preds

        return fuse_pred, prm_preds, fusion_preds


if __name__ == "__main__":
    model = FusionSegNet(num_cls=4).cuda()
    inp = torch.randn(1, 4, 80, 80, 80).cuda()
    mask = torch.tensor([[True, True, False, True]])
    out = model(inp, mask)
    print(out)
