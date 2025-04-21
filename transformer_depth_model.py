import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=6, patch_size=16, emb_size=768, num_patches=512):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
    
    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)  # shape: [B, emb_size, H', W']
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, emb_size]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_size]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + num_patches, emb_size]

        # Interpolate positional embedding dynamically
        if self.pos_embedding is None or self.pos_embedding.shape[1] != x.shape[1]:
            num_patches = H_p * W_p
            pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, x.shape[-1])).to(x.device)
            self.pos_embedding = pos_embed
        x = x + self.pos_embedding

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=768, depth=12, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads,
                                       dim_feedforward=int(emb_size * mlp_ratio),
                                       activation='gelu')
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerDepthEstimator(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_size=768, depth=12, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size=patch_size, emb_size=emb_size)
        self.transformer = TransformerEncoder(emb_size=emb_size, depth=depth, num_heads=num_heads)
        self.head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, patch_size * patch_size)  # predict a patch worth of pixels
        )
        self.patch_size = patch_size
        self.img_size = img_size

    def forward(self, left, right):
        x = torch.cat([left, right], dim=1) 
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, num_patches+1, emb_size]
        x = self.transformer(x)
        x = x[:, 1:]  # remove cls token
        x = self.head(x)  # [B, num_patches, patch_area]
        x = x.view(B, H // self.patch_size, W // self.patch_size, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, 1, H, W)
        return x  # [B, 1, H, W] depth map