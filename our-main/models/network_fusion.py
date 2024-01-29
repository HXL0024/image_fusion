import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

class MyFusion(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 bias=False,
                 ):
        super(MyFusion, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.dim = dim
        self.encoder = Restormer_Encoder()
        self.overlap_patch_embed_ir = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim)
        self.overlap_patch_embed_vis = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim)
        self.reduce_channel_ir = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.reduce_channel_vis = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.decoder = Restormer_Decoder()
        
    def forward(self,ir, vis, seg_ir, seg_vis):
        # 特征提取
        ir = ir[:, 0:1, :, :]
        vis = vis[:, 0:1, :, :]
        ir = self.encoder(ir)
        vis = self.encoder(vis)
        seg_ir = seg_ir[:, 0:1, :, :]
        seg_vis = seg_vis[:, 0:1, :, :]
        seg_ir = self.overlap_patch_embed_ir(seg_ir)
        seg_vis = self.overlap_patch_embed_vis(seg_vis)
        # 特征拼接
        fusion_ir = torch.cat((ir, seg_ir), dim=1)
        fusion_ir = self.reduce_channel_ir(fusion_ir)
        fusion_vis = torch.cat((vis, seg_vis), dim=1)
        fusion_vis = self.reduce_channel_vis(fusion_vis)
        # 重构
        x = self.decoder(fusion_ir, fusion_vis)
        return x

# Restormer  
class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 dim=64,
                 num_blocks=1,
                 heads=8,
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.restormer_encoder = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.RDB1 = RDB()
        self.RDB2 = RDB()
             
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.restormer_encoder(x)
        # 细节特征提取
        x = self.RDB1(x)
        x = self.RDB2(x)
        return x
    
# Restormer
class Restormer_Decoder(nn.Module):
    def __init__(self,
                 out_channels=1,
                 dim=64,
                 num_blocks=1,
                 heads=8,
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.restormer_decoder = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    
    def forward(self, fusion_ir, fusion_vis):
        fusion = torch.cat((fusion_ir, fusion_vis), dim=1)
        fusion = self.reduce_channel(fusion)
        fusion = self.restormer_decoder(fusion)
        fusion_img = self.output(fusion)
        return self.sigmoid(fusion_img)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)
    def forward(self, x):
        x = self.proj(x)
        return x
    
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
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    
class BiasFree_LayerNorm(nn.Module):
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
        return x / torch.sqrt(sigma+1e-5) * self.weight
    
import numbers
class WithBias_LayerNorm(nn.Module):
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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
      
    
class RDB_Conv(nn.Module):
    def __init__(self, in_ch=64, growth_rate=32):
        super(RDB_Conv, self).__init__()
        in_ch_ = in_ch
        growth_rate_ = growth_rate
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_ch_, growth_rate_, 3, padding=1, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), dim=1)
        return out
    
class RDB(nn.Module):
    def __init__(self, in_ch=64, growth_rate=32, num_layers=3):
        super(RDB, self).__init__()
        in_ch_ = in_ch
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(RDB_Conv(in_ch_, growth_rate))
            in_ch_ += growth_rate

        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0, stride=1)

    def forward(self, x):
        out = x
        for layer in self.layers:
            x = layer(x)
        x = self.conv(x)
        out = x + out
        return out