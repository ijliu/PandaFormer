from .build import MODEL_REGISTRY
import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg, Mlp
from functools import partial
from einops import rearrange
from .odconv import ODConv2d

_model_urls = {
    'crossvit_15_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_224.pth',
    'crossvit_15_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_224.pth',
    'crossvit_15_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_384.pth',
    'crossvit_18_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_224.pth',
    'crossvit_18_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_224.pth',
    'crossvit_18_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_384.pth',
    'crossvit_9_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_224.pth',
    'crossvit_9_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_dagger_224.pth',
    'crossvit_base_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_base_224.pth',
    'crossvit_small_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_small_224.pth',
    'crossvit_tiny_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_tiny_224.pth',
}

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans + 24, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans + 24, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            # self.proj = ODConv2d(in_chans + 1, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
            
        # for m in self.proj:
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_uniform(m.weight)
        #         nn.init.constant(m.bias, 0)
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, times):
        # x [2,3,8,240,240]
        B, C,T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        # print(x.shape)

        x = self.proj(x)
        
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        M = x.shape[-1]
        time_tokens = torch.ones((B,2,M)).to(x.device)
        for i in range(B):
            time_tokens[i][0] *= torch.sin(2 * torch.pi * times[i] / 24)
            time_tokens[i][1] *= torch.cos(2 * torch.pi * times[i] / 24)
        # time_tokens = None
        return x, time_tokens, W

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True, temporal=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.temporal = temporal

        if self.temporal:

            self.relative_position_bias_table = nn.Parameter(torch.zeros(15, num_heads))

            # print(self.relative_position_bias_table.shape)
            coords_t = torch.arange(8)
            # print(torch.meshgrid(coords_t, coords_t).shape)
            coords = torch.zeros((8, 8))
            for i in range(8):
                coords[i] = coords_t - i
            coords += 7

            self.coords = coords
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):

        # print(x.shape)

        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        # 对 q,k,v 进行改进
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.temporal:
            relative_position_bias = self.relative_position_bias_table[self.coords.view(-1).long()].view(8,8, -1)
            relative_position_bias = relative_position_bias.permute(2, 0 ,1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
           proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
              proj_drop=drop, temporal=True)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x):
        # print(len(input))
        B = x.shape[0]
        M = x.shape[-1]
        T = 8
        # (x,B,T,W) = input
        # num_spatial_tokens = (x.size(1) - 1) // T
        num_spatial_tokens = (x.size(1) - 1 -2) // T
        
        
        
        
        # H = num_spatial_tokens // W
        H = int(num_spatial_tokens ** 0.5)
        W = H

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,3:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,3:,:] + res_temporal
            
            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            
            ############## 绝对时间 ##################
            init_x_token = x[:,1,:].unsqueeze(1)
            x_token = init_x_token.repeat(1,T,1)
            x_token = rearrange(x_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            init_y_token = x[:,2,:].unsqueeze(1)
            y_token = init_y_token.repeat(1,T,1)
            y_token = rearrange(y_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            ##########################################
            
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            
            ############### 绝对时间 ####################
            xs = torch.cat((x_token, xs), 1)
            xs = torch.cat((y_token, xs), 1)
            ############################################
            
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            
            ################ 绝对时间 ######################
            x_token = res_spatial[:,1,:]
            y_token = res_spatial[:,2,:]
            x_token = rearrange(x_token, '(b t) m -> b t m',b=B,t=T)
            y_token = rearrange(y_token, '(b t) m -> b t m',b=B,t=T)
            x_token = torch.mean(x_token, 1, True)
            y_token = torch.mean(y_token, 1, True)
            ################################################
            
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            
            res_spatial = res_spatial[:,3:,:]
            
            
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt
            ## Mlp
            ############### 绝对时间 ###################
            # res = torch.cat((x_token, res), 1)
            res = torch.cat((x_token, res), 1)
            res = torch.cat((y_token, res), 1)
            x = torch.cat((init_x_token, x), 1)
            x = torch.cat((init_y_token, x), 1)
            ############################################
            
            
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape

        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
        #     proj_drop=drop, temporal=False)
        self.attn = CrossAttention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # depth = [1, 4, 0]
        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, 
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d+1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d+1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d+1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                       has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                   has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d+1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d+1) % num_branches]), act_layer(), nn.Linear(dim[(d+1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))


    def forward(self, x,T,W):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        # outs_b
        # [16, 401, 384]
        # [16, 197, 768]

        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # [2,1,384] -> [2,1,768]
        # [2,1,768] -> [2,1,384]
        
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            
            
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
    
        return outs

def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size,patches)]

class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=1000, 
                 embed_dim=(192, 384), depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, multi_conv=False):
        super().__init__()

        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size

        num_patches = _compute_num_patches(img_size, patch_size)
        # [400, 196]
        self.num_branches = len(patch_size)

        self.patch_embed = nn.ModuleList()
        if hybrid_backbone is None:
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.patch_embed.append(PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))
        # else:
        #     self.pos_embed = nn.ParameterList()
        #     from .t2t import T2T, get_sinusoid_encoding
        #     tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
        #     for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
        #         self.patch_embed.append(T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
        #         self.pos_embed.append(nn.Parameter(data=get_sinusoid_encoding(n_position=1 + num_patches[idx], d_hid=embed_dim[idx]), requires_grad=False))

        #     del self.pos_embed
        #     self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])

        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)])
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.time_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 8, embed_dim[i])) for i in range(self.num_branches)])
        self.time_drop = nn.Dropout(p=drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                                  attn_drop=attn_drop_rate, drop_path=dpr_,
                                  norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        self.head = nn.ModuleList([
            nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in range(self.num_branches)])
        
        ##################
        # hidd = [1024, 2048]
        # self.head = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(embed_dim[i] * 3, hidd[i]),
        #         nn.LayerNorm(hidd[i]),
        #         nn.GELU(),
        #         nn.Linear(hidd[i], num_classes),
        #     )
        #  for i in range(self.num_branches)])
        ##################

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
            if self.time_embed[i].requires_grad:
                trunc_normal_(self.time_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)

        self.apply(self._init_weights)
        
        # for i in range(self.num_branches):
        #     for m in self.head[i]:
        #         if isinstance(m, nn.Linear):
        #             nn.init.xavier_normal_(m.weight)
        #             nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.LayerNorm):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)

        self.B = []
        self.T = []

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, times):
        # B, C, H, W = x.shape
        # (B,3,8,224,224)
        B,C,T,H,W = x.shape
        self.B = B
        self.T = T
        self.W = []

        xs = []
        for i in range(self.num_branches):
            x_ = torch.nn.functional.interpolate(x, size=(T, self.img_size[i], self.img_size[i]), mode='trilinear') if H != self.img_size[i] else x
            tmp, time_tokens, w = self.patch_embed[i](x_, times)
            self.W.append(w)

            # cls_tokens = self.cls_token[i].expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token[i].expand(tmp.size(0), -1, -1)
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)

            ######## time embed ######
            cls_tokens = tmp[:B, 0, :].unsqueeze(1)
            tmp = tmp[:,1:]
            tmp = rearrange(tmp, '(b t) n m -> (b n) t m',b=B,t=T)
            tmp = tmp + self.time_embed[i]
            tmp = self.time_drop(tmp)
            tmp = rearrange(tmp, '(b n) t m -> b (n t) m',b=B,t=T)
            ####### 绝对时间 ########
            tmp = torch.cat((time_tokens, tmp), dim=1)
            ########################
            
            
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            ##########################

            xs.append(tmp)
        # blk_out = []
        for blk in self.blocks:
            xs = blk(xs, T, self.W)
            # blk_out.append(xs)       

        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        
        out = [x[:, 0] for x in xs]
        return out

        return rnt

    def forward(self, x, times):
        xs = self.forward_features(x, times)

        #     xs[i] = rearrange(xs[i], '(b t) m -> b t m', b=self.B,t=self.T)
        #     xs[i] = torch.mean(xs[i], dim=1)
        # for x in xs:
        #     print(x.shape)
        # print(self.head)
        # exit()

        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)
        return ce_logits

@MODEL_REGISTRY.register()
class crossvit_base_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(crossvit_base_224, self).__init__()
        self.pretrained=True
        # patch_size = 16
        self.model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], 
                              embed_dim=[384, 768], 
                              num_classes=5,
                              depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[12, 12], 
                              mlp_ratio=[4, 4, 1], 
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                              multi_conv=False,
                              **kwargs)
        self.model.default_cfg = _cfg()

        if self.pretrained:
            state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_base_224'], map_location='cpu')

            new_state_dict = state_dict.copy()
            for key in state_dict.keys():
                if "head" in key:
                    new_state_dict.pop(key)
                    print(f"size mismatch for {key}:")

            # new_state_dict = state_dict.copy()
            for key in state_dict:
                if 'blocks' in key and 'attn' in key:
                    new_key = key.replace('attn','temporal_attn')
                    if not new_key in state_dict:
                        new_state_dict[new_key] = state_dict[key]
                    else:
                        new_state_dict[new_key] = state_dict[new_key]
                if 'blocks' in key and 'norm1' in key:
                    new_key = key.replace('norm1','temporal_norm1')
                    if not new_key in state_dict:
                        new_state_dict[new_key] = state_dict[key]
                    else:
                        new_state_dict[new_key] = state_dict[new_key]
                        
                # if "patch_embed" in key and "proj.weight" in key:
                #     value = new_state_dict[key]
                #     m,c,h,w = value.shape
                #     new_state_dict[key] = torch.zeros((m,c+1,h,w))
                #     new_state_dict[key][:,0:3,...] = value
                    
                    # new_state_dict.pop(key)

            # state_dict = new_state_dict
            self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, x, times):
        # x [2,3,8,224,224]
        x = self.model(x, times)
        return x


@MODEL_REGISTRY.register()
class crossvit_18_dagger_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(crossvit_18_dagger_224, self).__init__()
        self.pretrained=True
        # patch_size = 16
        self.model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], 
                              embed_dim=[224, 448], 
                              depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], 
                              mlp_ratio=[3, 3, 1], 
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                              multi_conv=True, 
                              **kwargs)
        
        self.model.default_cfg = _cfg()

        if self.pretrained:
            state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_dagger_224'], map_location='cpu')

            new_state_dict = state_dict.copy()
            for key in state_dict.keys():
                if "head" in key:
                    new_state_dict.pop(key)
                    print(f"size mismatch for {key}:")

            # new_state_dict = state_dict.copy()
            for key in state_dict:
                if 'blocks' in key and 'attn' in key:
                    new_key = key.replace('attn','temporal_attn')
                    if not new_key in state_dict:
                        new_state_dict[new_key] = state_dict[key]
                    else:
                        new_state_dict[new_key] = state_dict[new_key]
                if 'blocks' in key and 'norm1' in key:
                    new_key = key.replace('norm1','temporal_norm1')
                    if not new_key in state_dict:
                        new_state_dict[new_key] = state_dict[key]
                    else:
                        new_state_dict[new_key] = state_dict[new_key]
                        
                # if "patch_embed" in key and "proj.weight" in key:
                #     value = new_state_dict[key]
                #     m,c,h,w = value.shape
                #     new_state_dict[key] = torch.zeros((m,c+1,h,w))
                #     new_state_dict[key][:,0:3,...] = value
                    
                    # new_state_dict.pop(key)

            # state_dict = new_state_dict
            self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        # x [2,3,8,224,224]
        x = self.model(x)
        return x


@MODEL_REGISTRY.register()
class crossvit_18_dagger_384(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(crossvit_18_dagger_384, self).__init__()
        self.pretrained=True
        # patch_size = 16
        self.model = VisionTransformer(img_size=[408, 384],
                              patch_size=[12, 16], 
                              embed_dim=[224, 448], 
                              depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], 
                              mlp_ratio=[3, 3, 1], 
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                              multi_conv=True, 
                              **kwargs)
        self.model.default_cfg = _cfg()

        if self.pretrained:
            state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_dagger_384'], map_location='cpu')

            new_state_dict = state_dict.copy()
            for key in state_dict.keys():
                if "head" in key:
                    new_state_dict.pop(key)
                    print(f"size mismatch for {key}:")

            # new_state_dict = state_dict.copy()
            for key in state_dict:
                if 'blocks' in key and 'attn' in key:
                    new_key = key.replace('attn','temporal_attn')
                    if not new_key in state_dict:
                        new_state_dict[new_key] = state_dict[key]
                    else:
                        new_state_dict[new_key] = state_dict[new_key]
                if 'blocks' in key and 'norm1' in key:
                    new_key = key.replace('norm1','temporal_norm1')
                    if not new_key in state_dict:
                        new_state_dict[new_key] = state_dict[key]
                    else:
                        new_state_dict[new_key] = state_dict[new_key]
                        
                # if "patch_embed" in key and "proj.weight" in key:
                #     value = new_state_dict[key]
                #     m,c,h,w = value.shape
                #     new_state_dict[key] = torch.zeros((m,c+1,h,w))
                #     new_state_dict[key][:,0:3,...] = value
                    
                    # new_state_dict.pop(key)

            # state_dict = new_state_dict
            self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        # x [2,3,8,224,224]
        x = self.model(x)
        return x
    
