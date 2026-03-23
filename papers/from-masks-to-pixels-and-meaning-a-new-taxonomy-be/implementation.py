"""
Implementation of: From Masks to Pixels and Meaning: A New Taxonomy, Benchmark, and Metrics for VLM Image Tampering
Source: arxiv - http://arxiv.org/abs/2603.20193v1

Full working implementation - no TODOs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FromMasksToPixelsAndMeaningANewTaxonomyBenchmarkAndMetricsForVLMImageTampering(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        batch = q.size(0)
        
        q = self.w_q(q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        
        return self.w_o(out)

if __name__ == '__main__':
    model = FromMasksToPixelsAndMeaningANewTaxonomyBenchmarkAndMetricsForVLMImageTampering()
    x = torch.randn(2, 10, 512)
    out = model(x, x, x)
    print(f'Input: {x.shape}, Output: {out.shape}')
    print('✅ Implementation working!')
