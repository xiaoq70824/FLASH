import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange


#-------------------------------SimpleGate-------------------------------

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        # If the sizes of x1 and x2 are different, adjust them to ensure they have the same size.
        if x1.size(1) != x2.size(1):
            # Take the smaller size
            min_channels = min(x1.size(1), x2.size(1))
            x1 = x1[:, :min_channels, :, :]
            x2 = x2[:, :min_channels, :, :]
        return x1 * x2


#-------------------------------LayerNorm-------------------------------

def to_3d(x):
    # Reshape the 4D tensor (batch, channel, height, width) into a 3D tensor (batch, height*width, channel)
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    # Reshape the 3D tensor (batch, height*width, channel) back into a 4D tensor (batch, channel, height, width)
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    """
        Layer Normalization without Bias
        Compute the variance of the input and use it to normalize the input data
    """

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """
    Layer Normalization with Bias
    Compute the mean and variance of the input, and use them to normalize the input data
    """

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


#-------------------------------SkipPatchEmbed-------------------------------
class SkipPatchEmbed(nn.Module):
    def __init__(self, in_c=3, dim=48, bias=False):
        super(SkipPatchEmbed, self).__init__()

        self.proj = nn.Sequential(
            nn.AvgPool2d(2, stride=2, padding=0, ceil_mode=False, count_include_pad=True),
            nn.Conv2d(in_c, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )

    def forward(self, x):
        return self.proj(x)


#-------------------------------No-Activation Frequency Feedforward-------------------------------
class NAFF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(NAFF, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = x1 * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        h_n = (self.patch_size - h % self.patch_size) % self.patch_size
        w_n = (self.patch_size - w % self.patch_size) % self.patch_size

        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)

        x = x[:, :, :h, :w]

        return x


#-------------------------------No-Activation Frequency Block-------------------------------
class NAFB(nn.Module):
    def __init__(self, dim, DW_Expand=2, FFN_Expand=2.66, drop_out_rate=0., LayerNorm_type='WithBias', bias=False):
        super().__init__()
        # 替换NAFNet的LayerNorm2d为AdaIR的LayerNorm
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        dw_channel = int(dim * DW_Expand)
        # 确保 dw_channel 是偶数
        dw_channel = dw_channel + (dw_channel % 2)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()


        self.naff = NAFF(dim=dim, ffn_expansion_factor=FFN_Expand, bias=bias)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.naff(self.norm2(y))

        x = self.dropout2(x)

        return y + x * self.gamma


#-------------------------------Resizing modules-------------------------------
class Downsample(nn.Module):
    """
        Downsampling Module
        Reduce the spatial size of the feature map to half of the original,
        while expanding the number of channels to twice the original
    """

    def __init__(self, n_feat):
        # Initialize the parent class (nn.Module)
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """
        Upsampling Module
        Increase the spatial size of the feature map to twice the original,Increase the spatial size of the feature map to twice the original,
        while reducing the number of channels to half of the original
    """

    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


#-------------------------------Channel-Wise Cross Attention (CA)-------------------------------
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out

#-------------------------------Overlapped image patch embedding with 3x3 Conv-------------------------------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


#-------------------------------FDHA: FREQUENCY-DRIVEN HISTOGRAM ATTENTION-------------------------------
class FDHA(nn.Module):
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(FDHA, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.factor = 8  # Set the number factor of bins

        # The initial convolutional layer processes the input
        self.conv_in = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        # Feature extraction layer
        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)

        # Output Projection Layer
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Cross Attention Processing
        self.ca = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)

        # Learning Parameter
        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1e-6)
        return logit

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor

        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"

        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)

        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)

        return out

    def forward(self, x, y):
        b, c, h, w = y.shape

        # 1. Resize the spatial size of x to match that of y
        if x.shape[2:] != y.shape[2:]:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        # 2. Adjust the number of channels of x via 3x3 convolution
        x_conv = self.conv_in(x)

        # 3. Store the original input for residual connection
        y_orig = y.clone()

        # 4. First half channel processing - Using FFT (Fast Fourier Transform) and frequency-domain sorting
        half_c = c // 2
        y_half = y[:, :half_c]
        fft_half = torch.fft.fft2(y_half, norm='forward', dim=(-2, -1))
        fft_abs_half = torch.abs(fft_half)

        # Sort along the height dimension
        _, idx_h = fft_abs_half.sort(dim=2)
        y_half_sorted = torch.gather(y_half, dim=2, index=idx_h)

        # Sort along the width dimension
        _, idx_w = fft_abs_half.sort(dim=3)
        y_half_sorted = torch.gather(y_half_sorted, dim=3, index=idx_w)

        # Replace the sorted results back into the first half channels of y
        y = y.clone()
        y[:, :half_c] = y_half_sorted

        # 5. Extract QKV (Query, Key, Value) features
        qkv = self.qkv_dwconv(self.qkv(y))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        # 6. Perform frequency-domain transformation and sorting on v
        v_fft = torch.fft.fft2(v, norm='forward', dim=(-2, -1))
        v_fft_abs = torch.abs(v_fft)
        v_fft_abs_flat = v_fft_abs.view(b, v.size(1), -1)
        _, idx = v_fft_abs_flat.sort(dim=-1)  # Sort using frequency-domain amplitude

        # 7. Rearrange q1, k1, q2, k2 according to the frequency-domain sorting indices of v
        q1_flat = q1.view(b, q1.size(1), -1)
        k1_flat = k1.view(b, k1.size(1), -1)
        q2_flat = q2.view(b, q2.size(1), -1)
        k2_flat = k2.view(b, k2.size(1), -1)
        v_flat = v.view(b, v.size(1), -1)

        q1_sorted = torch.gather(q1_flat, dim=2, index=idx)
        k1_sorted = torch.gather(k1_flat, dim=2, index=idx)
        q2_sorted = torch.gather(q2_flat, dim=2, index=idx)
        k2_sorted = torch.gather(k2_flat, dim=2, index=idx)
        v_sorted = torch.gather(v_flat, dim=2, index=idx)

        # 8. Calculate BHR and FHR
        out1 = self.reshape_attn(q1_sorted, k1_sorted, v_sorted, True)  # BHR
        out2 = self.reshape_attn(q2_sorted, k2_sorted, v_sorted, False)  # FHR

        # 9. Restore the original order using scatter operation
        out1 = torch.scatter(torch.zeros_like(q1_flat), 2, idx, out1).view(b, q1.size(1), h, w)
        out2 = torch.scatter(torch.zeros_like(q2_flat), 2, idx, out2).view(b, q2.size(1), h, w)

        # 10. Merge the results of BHR and FHR
        out = out1 * out2
        out = self.project_out(out)

        # 11. Apply reverse sorting only to the first half channels
        out_replace = out[:, :half_c]
        out_replace = torch.scatter(torch.zeros_like(out_replace), -1, idx_w, out_replace)
        out_replace = torch.scatter(torch.zeros_like(out_replace), -2, idx_h, out_replace)
        out[:, :half_c] = out_replace

        # 12. Calculate CA (Cross Attention) with x_conv, where x_conv serves as K (Key) and V (Value), and output serves as Q (Query)
        agg = self.ca(out, x_conv)

        # 13. Weighted residual connection
        return agg * self.para1 + y_orig * self.para2


#-------------------------------FLASH-------------------------------

class FLASH(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 drop_out_rate=0.
                 ):
        super(FLASH, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.fm_enc_level1 = FDHA(dim, num_heads=heads[0], bias=bias)
        self.fm_enc_level2 = FDHA(dim * 2 ** 1, num_heads=heads[1], bias=bias)
        self.fm_enc_level3 = FDHA(dim * 2 ** 2, num_heads=heads[2], bias=bias)
        self.fm_latent = FDHA(dim * 2 ** 3, num_heads=heads[3], bias=bias)

        self.fm_dec_level3 = FDHA(dim * 2 ** 2, num_heads=heads[2], bias=bias)
        self.fm_dec_level2 = FDHA(dim * 2 ** 1, num_heads=heads[1], bias=bias)
        self.fm_dec_level1 = FDHA(dim * 2 ** 1, num_heads=heads[0], bias=bias)
        self.fm_refinement = FDHA(dim * 2 ** 1, num_heads=heads[0], bias=bias)

        self.skip_patch_embed1 = SkipPatchEmbed(inp_channels, dim)
        self.skip_patch_embed2 = SkipPatchEmbed(dim, dim * 2)
        self.skip_patch_embed3 = SkipPatchEmbed(dim * 2, dim * 4)

        self.reduce_chan_level_1 = nn.Conv2d(int(dim * 2 ** 1) + dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level_2 = nn.Conv2d(int(dim * 2 ** 2) + dim * 2, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level_3 = nn.Conv2d(int(dim * 2 ** 3) + dim * 4, int(dim * 2 ** 3), kernel_size=1, bias=bias)

        self.encoder_level1 = nn.Sequential(*[
            NAFB(dim=dim, DW_Expand=2, FFN_Expand=ffn_expansion_factor, drop_out_rate=drop_out_rate,
                 LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[
            NAFB(dim=int(dim * 2 ** 1), DW_Expand=2, FFN_Expand=ffn_expansion_factor,
                 drop_out_rate=drop_out_rate,
                 LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[
            NAFB(dim=int(dim * 2 ** 2), DW_Expand=2, FFN_Expand=ffn_expansion_factor,
                 drop_out_rate=drop_out_rate,
                 LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            NAFB(dim=int(dim * 2 ** 3), DW_Expand=2, FFN_Expand=ffn_expansion_factor,
                 drop_out_rate=drop_out_rate,
                 LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[
            NAFB(dim=int(dim * 2 ** 2), DW_Expand=2, FFN_Expand=ffn_expansion_factor,
                 drop_out_rate=drop_out_rate,
                 LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            NAFB(dim=int(dim * 2 ** 1), DW_Expand=2, FFN_Expand=ffn_expansion_factor,
                 drop_out_rate=drop_out_rate,
                 LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            NAFB(dim=int(dim * 2 ** 1), DW_Expand=2, FFN_Expand=ffn_expansion_factor,
                 drop_out_rate=drop_out_rate,
                 LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            NAFB(dim=int(dim * 2 ** 1), DW_Expand=2, FFN_Expand=ffn_expansion_factor,
                 drop_out_rate=drop_out_rate,
                 LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, noise_emb=None):
        # Initial feature embedding: Convert input image to initial feature representation
        inp_enc_level1 = self.patch_embed(inp_img)

        # Level 1 encoder: Process initial features and extract low-level information
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # Apply FDHA module: Enhance feature representation using Frequency-Driven Histogram Attention
        out_enc_level1 = self.fm_enc_level1(inp_img, out_enc_level1)

        # Level 1 downsampling: Reduce spatial resolution while increasing channel dimensions
        inp_enc_level2 = self.down1_2(out_enc_level1)
        # Create level 1 skip connection: Extract skip features from original image for later fusion
        skip_enc_level1 = self.skip_patch_embed1(inp_img)
        # Feature fusion: Concatenate downsampled features with skip features and reduce channels
        inp_enc_level2 = torch.cat([inp_enc_level2, skip_enc_level1], 1)
        inp_enc_level2 = self.reduce_chan_level_1(inp_enc_level2)

        # Level 2 encoder: Process higher-level feature representations
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        # Apply FDHA module: Further enhance mid-level feature representation capability
        out_enc_level2 = self.fm_enc_level2(inp_img, out_enc_level2)

        # Level 2 downsampling: Continue reducing resolution and increasing receptive field
        inp_enc_level3 = self.down2_3(out_enc_level2)
        # Create level 2 skip connection: Build level 2 skip features based on level 1 skip features
        skip_enc_level2 = self.skip_patch_embed2(skip_enc_level1)
        # Feature fusion: Integrate multi-scale information
        inp_enc_level3 = torch.cat([inp_enc_level3, skip_enc_level2], 1)
        inp_enc_level3 = self.reduce_chan_level_2(inp_enc_level3)

        # Level 3 encoder: Extract high-level semantic features
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        # Apply FDHA module: Enhance discriminative capability of high-level features
        out_enc_level3 = self.fm_enc_level3(inp_img, out_enc_level3)

        # Level 3 downsampling: Reach minimum spatial resolution
        inp_enc_level4 = self.down3_4(out_enc_level3)
        # Create level 3 skip connection: Build deepest skip connection
        skip_enc_level3 = self.skip_patch_embed3(skip_enc_level2)
        # Feature fusion: Prepare input features for bottleneck layer
        inp_enc_level4 = torch.cat([inp_enc_level4, skip_enc_level3], 1)
        inp_enc_level4 = self.reduce_chan_level_3(inp_enc_level4)

        # Bottleneck processing: Perform feature transformation and semantic understanding at deepest level
        latent = self.latent(inp_enc_level4)
        # Apply FDHA module: Enhance global feature representation at bottleneck layer
        latent = self.fm_latent(inp_img, latent)

        # === Decoder Section: Gradually restore spatial resolution and fuse encoder features ===

        # Level 3 decoder: Begin upsampling reconstruction process
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        # Apply FDHA module: Enhance feature representation during decoding process
        out_dec_level3 = self.fm_dec_level3(inp_img, out_dec_level3)

        # Level 2 decoder: Continue upsampling and feature reconstruction
        inp_dec_level2 = self.up3_2(out_dec_level3)
        ## Skip connection fusion: Maintain detail information propagation
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        # Apply FDHA module: Optimize mid-level decoding features
        out_dec_level2 = self.fm_dec_level2(inp_img, out_dec_level2)

        # Level 1 decoder: Restore to near-original resolution
        inp_dec_level1 = self.up2_1(out_dec_level2)
        # Skip connection fusion: Integrate final encoder features
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        # Apply FDHA module: Final feature optimization
        out_dec_level1 = self.fm_dec_level1(inp_img, out_dec_level1)

        # Feature refinement: Further improve output quality
        out_dec_level1 = self.refinement(out_dec_level1)
        # Apply FDHA module: Final feature enhancement processing
        out_dec_level1 = self.fm_refinement(inp_img, out_dec_level1)
        # Output layer: Generate final result with residual connection to preserve input information
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


if __name__ == "__main__":
    # Model instantiation
    model = FLASH(
        inp_channels=3,
        out_channels=3,
        dim=32,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        drop_out_rate=0.
    )

    # Parameter count statistics (in millions)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # GPU forward propagation speed test
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(2, 3, 256, 256).cuda()

        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)

        # Speed test
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_time.record()
            _ = model(x)
            end_time.record()
            torch.cuda.synchronize()

        print(f"GPU forward propagation speed: {start_time.elapsed_time(end_time):.2f}ms")
    else:
        print("CUDA unavailable, unable to test GPU speed")