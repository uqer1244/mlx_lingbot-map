#!/usr/bin/env python3
"""
LingBot-MAP MLX Model Architecture for Apple Silicon (Metal).
"""
import os
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any

# ==============================================================================
# 1. Geometry & Interpolation Helpers (Pure MLX)
# ==============================================================================

def bilinear_resize_mlx(img: mx.array, H_out: int, W_out: int) -> mx.array:
    """Pure MLX Vectorized Bilinear Interpolation."""
    H_in, W_in, C = img.shape
    
    ys = mx.arange(H_out, dtype=mx.float32)
    xs = mx.arange(W_out, dtype=mx.float32)
    
    xs_in = xs * ((W_in - 1) / max(1, W_out - 1))
    ys_in = ys * ((H_in - 1) / max(1, H_out - 1))
    
    x0 = mx.floor(xs_in).astype(mx.int32)
    x1 = mx.minimum(x0 + 1, W_in - 1)
    
    y0 = mx.floor(ys_in).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, H_in - 1)
    
    wx = (xs_in - x0).reshape(1, W_out, 1)
    wy = (ys_in - y0).reshape(H_out, 1, 1)
    
    img_y0 = mx.take(img, y0, axis=0)
    img_y1 = mx.take(img, y1, axis=0)
    
    c00 = mx.take(img_y0, x0, axis=1)
    c10 = mx.take(img_y0, x1, axis=1)
    c01 = mx.take(img_y1, x0, axis=1)
    c11 = mx.take(img_y1, x1, axis=1)
    
    top = (1.0 - wx) * c00 + wx * c10
    bottom = (1.0 - wx) * c01 + wx * c11
    return (1.0 - wy) * top + wy * bottom

def batched_bilinear_resize(x: mx.array, H_out: int, W_out: int) -> mx.array:
    """Apply bilinear_resize_mlx over batch dimension [B, H_in, W_in, C]."""
    B = x.shape[0]
    outs = [bilinear_resize_mlx(x[b], H_out, W_out) for b in range(B)]
    return mx.stack(outs, axis=0)

def create_uv_grid_mlx(width: int, height: int, aspect_ratio: float = None) -> mx.array:
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    
    x_coords = mx.linspace(left_x, right_x, num=width)
    y_coords = mx.linspace(top_y, bottom_y, num=height)
    
    uu, vv = mx.meshgrid(x_coords, y_coords, indexing="xy")
    return mx.stack([uu, vv], axis=-1)

def make_sincos_pos_embed_mlx(embed_dim: int, pos: mx.array, omega_0: float = 100.0) -> mx.array:
    assert embed_dim % 2 == 0
    omega = mx.arange(embed_dim // 2, dtype=mx.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (omega_0 ** omega)
    
    pos = pos.reshape(-1)
    out = pos[:, None] * omega[None, :]
    
    emb_sin = mx.sin(out)
    emb_cos = mx.cos(out)
    return mx.concatenate([emb_sin, emb_cos], axis=1)

def position_grid_to_embed_mlx(pos_grid: mx.array, embed_dim: int, omega_0: float = 100.0) -> mx.array:
    H, W, grid_dim = pos_grid.shape
    pos_flat = pos_grid.reshape(-1, grid_dim)
    
    emb_x = make_sincos_pos_embed_mlx(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed_mlx(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)
    
    emb = mx.concatenate([emb_x, emb_y], axis=-1)
    return emb.reshape(H, W, embed_dim)

# ==============================================================================
# 2. DPT Head Modules (MLX Version)
# ==============================================================================

class MLXResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

    def __call__(self, x):
        out = nn.relu(x)
        out = self.conv1(out)
        out = nn.relu(out)
        out = self.conv2(out)
        return x + out

class MLXFeatureFusionBlock(nn.Module):
    def __init__(self, features: int, has_residual: bool = True):
        super().__init__()
        self.has_residual = has_residual
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        if has_residual:
            self.resConfUnit1 = MLXResidualConvUnit(features)
        self.resConfUnit2 = MLXResidualConvUnit(features)

    def __call__(self, x, residual=None, size=None):
        output = x
        if self.has_residual and residual is not None:
            res = self.resConfUnit1(residual)
            output = output + res
        output = self.resConfUnit2(output)
        
        if size is not None:
            output = batched_bilinear_resize(output, size[0], size[1])
        else:
            output = batched_bilinear_resize(output, output.shape[1] * 2, output.shape[2] * 2)
            
        output = self.out_conv(output)
        return output

class MLXScratch(nn.Module):
    def __init__(self, in_shape: List[int], out_shape: int):
        super().__init__()
        self.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.refinenet1 = MLXFeatureFusionBlock(out_shape, has_residual=True)
        self.refinenet2 = MLXFeatureFusionBlock(out_shape, has_residual=True)
        self.refinenet3 = MLXFeatureFusionBlock(out_shape, has_residual=True)
        self.refinenet4 = MLXFeatureFusionBlock(out_shape, has_residual=False)
        
        self.output_conv1 = nn.Conv2d(out_shape, out_shape // 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.output_conv2_0 = nn.Conv2d(out_shape // 2, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.output_conv2_2 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=True)

class MLXDPTHead(nn.Module):
    def __init__(self, dim_in: int, output_dim: int = 2, features: int = 256, out_channels: List[int] = [256, 512, 1024, 1024]):
        super().__init__()
        self.patch_size = 14
        self.norm = nn.LayerNorm(dim_in)
        
        self.projects = [
            nn.Conv2d(dim_in, out_channels[0], kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(dim_in, out_channels[1], kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(dim_in, out_channels[2], kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(dim_in, out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)
        ]
        
        self.resize_layers = [
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0, bias=True),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0, bias=True),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1, bias=True)
        ]
        
        self.scratch = MLXScratch(out_channels, features)

    def _apply_pos_embed(self, x: mx.array, W: int, H: int, ratio: float = 0.1) -> mx.array:
        H_p, W_p = x.shape[1], x.shape[2]
        pos_grid = create_uv_grid_mlx(W_p, H_p, aspect_ratio=float(W)/H)
        pos_embed = position_grid_to_embed_mlx(pos_grid, x.shape[3])
        return x + (pos_embed * ratio)

    def __call__(self, aggregated_tokens_list: List[mx.array], images: mx.array, patch_start_idx: int) -> Tuple[mx.array, mx.array]:
        B, S, H, W, _ = images.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        out_features = []
        for idx in range(4):
            x = aggregated_tokens_list[idx][:, :, patch_start_idx:]
            C_in = x.shape[-1]
            x = x.reshape(B * S, patch_h * patch_w, C_in)
            
            x = self.norm(x)
            x = x.reshape(B * S, patch_h, patch_w, C_in)
            
            x = self.projects[idx](x)
            x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[idx](x)
            
            out_features.append(x)
            
        layer_1, layer_2, layer_3, layer_4 = out_features
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[1:3])
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[1:3])
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[1:3])
        out = self.scratch.refinenet1(out, layer_1_rn)
        
        out = self.scratch.output_conv1(out)
        out = batched_bilinear_resize(out, H, W)
        out = self._apply_pos_embed(out, W, H)
        
        out = self.scratch.output_conv2_0(out)
        out = nn.relu(out)
        out = self.scratch.output_conv2_2(out)
        
        depth_raw = out[:, :, :, 0:1]
        conf_raw = out[:, :, :, 1:2]
        
        depth = mx.exp(depth_raw)
        depth_conf = 1.0 + mx.exp(conf_raw)
        
        depth = depth.reshape(B, S, H, W, 1)
        depth_conf = depth_conf.reshape(B, S, H, W, 1)
        return depth, depth_conf

# ==============================================================================
# 3. Aggregator & Main GCTStream (MLX Version)
# ==============================================================================

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = mx.ones((dim,)) * init_values

    def __call__(self, x):
        return x * self.gamma

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, proj_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def __call__(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

class PatchEmbedSub(nn.Module):
    def __init__(self, patch_size=14, embed_dim=1024):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x):
        x = self.proj(x)
        B, Hp, Wp, C = x.shape
        return x.reshape(B, Hp * Wp, C)

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16):
        super().__init__()
        self.patch_embed = PatchEmbedSub()
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, 1370, embed_dim))
        self.register_tokens = mx.zeros((1, 4, embed_dim))
        self.mask_token = mx.zeros((1, embed_dim))
        self.blocks = [Block(embed_dim, num_heads) for _ in range(24)]

class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, proj_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.q_norm = nn.LayerNorm(dim // num_heads)
        self.k_norm = nn.LayerNorm(dim // num_heads)

    def __call__(self, x, causal_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if causal_mask is not None:
            attn = attn + causal_mask
        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)

class GlobalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GlobalAttention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def __call__(self, x, causal_mask=None):
        x = x + self.ls1(self.attn(self.norm1(x), causal_mask=causal_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

def interpolate_pos_encoding_mlx(pos_embed: mx.array, H_p: int, W_p: int) -> mx.array:
    cls_pos = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]
    dim = pos_embed.shape[-1]
    M = int(round(patch_pos.shape[1] ** 0.5))
    if (H_p, W_p) == (M, M):
        return pos_embed
    patch_pos_2d = patch_pos.reshape(1, M, M, dim)
    patch_pos_resized = batched_bilinear_resize(patch_pos_2d, H_p, W_p)
    patch_pos_flat = patch_pos_resized.reshape(1, H_p * W_p, dim)
    return mx.concatenate([cls_pos, patch_pos_flat], axis=1)

class Aggregator(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16):
        super().__init__()
        self.camera_token = mx.zeros((1, 2, 1, embed_dim))
        self.register_token = mx.zeros((1, 2, 4, embed_dim))
        self.scale_token = mx.zeros((1, 2, 1, embed_dim))
        self.patch_embed = PatchEmbed(embed_dim, num_heads)
        self.global_blocks = [GlobalBlock(embed_dim, num_heads) for _ in range(24)]

    def __call__(self, x):
        H_img, W_img = x.shape[1], x.shape[2]
        H_p, W_p = H_img // 14, W_img // 14
        
        x_proj = self.patch_embed.patch_embed(x)
        B_flat, N_patches, C_embed = x_proj.shape
        
        cls_t = mx.broadcast_to(self.patch_embed.cls_token, (B_flat, 1, C_embed))
        x_tokens = mx.concatenate([cls_t, x_proj], axis=1)
        
        pos_embed = interpolate_pos_encoding_mlx(self.patch_embed.pos_embed, H_p, W_p)
        x_tokens = x_tokens + pos_embed
        
        reg_t = mx.broadcast_to(self.patch_embed.register_tokens, (B_flat, 4, C_embed))
        x_tokens = mx.concatenate([x_tokens[:, :1, :], reg_t, x_tokens[:, 1:, :]], axis=1)
        
        out_list = []
        for idx in range(24):
            x_tokens = self.patch_embed.blocks[idx](x_tokens)
            frame_inter = x_tokens
            
            x_tokens = self.global_blocks[idx](x_tokens)
            global_inter = x_tokens
            
            if idx in [4, 11, 17, 23]:
                concat_inter = mx.concatenate([frame_inter, global_inter], axis=-1)
                out_list.append(concat_inter)
                
        return out_list

class MLXGCTStream(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.aggregator = Aggregator(embed_dim)
        self.depth_head = MLXDPTHead(dim_in=2 * embed_dim, output_dim=2)

    def __call__(self, images: mx.array) -> Dict[str, mx.array]:
        B, S, H, W, C = images.shape
        x = images.reshape(B * S, H, W, C)
        
        aggregated_tokens_list = self.aggregator(x)
        
        reshaped_tokens_list = []
        for feat in aggregated_tokens_list:
            reshaped_tokens_list.append(feat.reshape(B, S, feat.shape[1], feat.shape[2]))
            
        depth, depth_conf = self.depth_head(reshaped_tokens_list, images, patch_start_idx=5)
        return {"depth": depth, "confidence": depth_conf}

    def load_weights(self, weights_path: str):
        print(f"Loading MLX safetensors weights from {weights_path}...")
        weights = mx.load(weights_path)
        
        # Build state dict key mapping for MLX module hierarchy
        updated_dict = {}
        for k, v in weights.items():
            key = k
            key = key.replace("depth_head.scratch.output_conv2.0.", "depth_head.scratch.output_conv2_0.")
            key = key.replace("depth_head.scratch.output_conv2.2.", "depth_head.scratch.output_conv2_2.")
            updated_dict[key] = v
            
        self.load_weights_dict(updated_dict)
        print("MLX weights loaded successfully!")

    def load_weights_dict(self, weights_dict: Dict[str, mx.array]):
        # Custom assigner for nested MLX module parameters
        def _assign_recursive(mod, prefix=""):
            if isinstance(mod, list):
                for idx, item in enumerate(mod):
                    _assign_recursive(item, f"{prefix}{idx}.")
                return
            elif isinstance(mod, dict):
                for k, item in mod.items():
                    _assign_recursive(item, f"{prefix}{k}.")
                return

            if hasattr(mod, "children"):
                for k, child in mod.children().items():
                    _assign_recursive(child, f"{prefix}{k}.")

            if hasattr(mod, "parameters"):
                for k, val in mod.parameters().items():
                    full_k = f"{prefix}{k}"
                    if full_k in weights_dict:
                        setattr(mod, k, weights_dict[full_k])

        _assign_recursive(self)
