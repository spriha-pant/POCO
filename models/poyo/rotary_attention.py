print("ROTARY FILE LOADED")
import logging
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

xops = None # Disable xformers (not supported on SM 12.0)

from models.poyo.rotary_embedding import apply_rotary_pos_emb

class RotaryCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
        use_memory_efficient_attn: bool =True,
    ):
        super().__init__()

        if use_memory_efficient_attn and xops is None:
            logging.warning(
                "xformers is not installed, falling back to default attention"
            )
            use_memory_efficient_attn = False

        inner_dim = dim_head * heads
        context_dim = context_dim or dim
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value
        # self.using_memory_efficient_attn = use_memory_efficient_attn
        self.using_memory_efficient_attn = False # Disable xformers (not supported on SM 12.0)

        # build networks
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        print("Using torch fallback")

    def forward(
        self,
        x_query,
        x_context,
        rotary_time_emb_query,
        rotary_time_emb_context,
        *,
        context_mask=None,
        query_seqlen=None,
        context_seqlen=None,
    ):

        # normalize and project to q, k, v
        x_query = self.norm(x_query)
        x_context = self.norm_context(x_context)

        q = self.to_q(x_query)
        k, v = self.to_kv(x_context).chunk(2, dim=-1)

        if False:
            if context_mask is not None:
                raise NotImplementedError(
                    f"Got non-None `attn_mask`. "
                    f"This implementation with memory efficient attention only works "
                    f"with `x_seqlen` for handling unequal sample lengths. Traditional "
                    f"padding approach is supported with normal non-memory efficient "
                    f"attention."
                )

            if query_seqlen is None or context_seqlen is None:
                raise ValueError(
                    f"Both `query_seqlen` and `context_seqlen` must be valid "
                    f"sequence lengths."
                )
            
            print("ENTERED rotary_memory_efficient_attention")

            out = rotary_memory_efficient_attention(
                q=q, k=k, v=v,
                rotary_time_emb_q=rotary_time_emb_query, 
                rotary_time_emb_kv=rotary_time_emb_context,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                q_seqlen=query_seqlen,
                kv_seqlen=context_seqlen,
            )

        else:
            if not self.using_memory_efficient_attn:
                q_seqlen = None
                kv_seqlen = None

            out = rotary_default_attention(
                q=q, k=k, v=v,
                rotary_time_emb_q=rotary_time_emb_query, 
                rotary_time_emb_kv=rotary_time_emb_context,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                kv_mask=context_mask,
            )
        
        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
        use_memory_efficient_attn: bool = True,
    ):
        super().__init__()

        if use_memory_efficient_attn and xops is None:
            logging.warning(
                "xformers is not installed, falling back to default attention"
            )
            use_memory_efficient_attn = False

        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        # self.using_memory_efficient_attn = use_memory_efficient_attn
        self.using_memory_efficient_attn = False # Disable xformers (not supported on SM 12.0)
        self.rotate_value = rotate_value

        # build networks
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        print("Using torch fallback")

    def forward(
        self, 
        x, 
        rotary_time_emb, 
        *,
        x_mask=None,
        x_seqlen=None,
        
    ):

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        if self.using_memory_efficient_attn:
            out = self.standard_attention(
                q, k, v,
                query_seqlen=query_seqlen,
                context_seqlen=context_seqlen,
                context_mask=context_mask 
)

        else:
            if not self.using_memory_efficient_attn:
                q_seqlen = None
                kv_seqlen = None

            out = rotary_default_attention(
                q=q, k=k, v=v,
                rotary_time_emb_q=rotary_time_emb,
                rotary_time_emb_kv=rotary_time_emb,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                kv_mask=x_mask,
            )
        
        out = self.to_out(out)
        return out


def rotary_default_attention(
    *,
    q, # (b, n_q, (h d), )
    k, # (b, n_kv, (h d), )
    v, # (b, n_kv, (h d), )
    rotary_time_emb_q, # (b, n_q, d)
    rotary_time_emb_kv, # (b, n_kv, d)
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
    kv_mask=None, # (b, n_kv)
): # Output: (b, n, (h d), )
    r"""Wraps the default attention implementation with rotary embedding application.
    """

    if q.dim() == 2:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        added_batch = True
    else:
        added_batch = False

    if q.ndim == 4:
        # already in [B, H, N, D]
        pass
    else:
        q = rearrange(q, "b n (h d) -> b h n d", h=num_heads)

    if k.ndim == 4:
        pass
    else:
        k = rearrange(k, "b n (h d) -> b h n d", h=num_heads)

    if v.ndim == 4:
        pass
    else:
        v = rearrange(v, "b n (h d) -> b h n d", h=num_heads)

    # # default attention expects shape b h n d
    # q = rearrange(q, "b n (h d) -> b h n d", h=num_heads)
    # k = rearrange(k, "b n (h d) -> b h n d", h=num_heads)
    # v = rearrange(v, "b n (h d) -> b h n d", h=num_heads)

    if added_batch:
        rotary_time_emb_q = rotary_time_emb_q.unsqueeze(0)
        rotary_time_emb_kv = rotary_time_emb_kv.unsqueeze(0)

    # attention mask
    if kv_mask is not None:
        kv_mask = rearrange(kv_mask, "b n -> b () () n") 

    if q.ndim == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        added_head = True
    else:
        added_head = False

    # perform attention, by default will use the optimal attention implementation
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask = None, dropout_p=dropout_p,
    )

    if added_head:
        out = out.squeeze(1)

    if added_batch:
        out = out.squeeze(0)

    # return (b, n, (h d), )
    if out.ndim == 4:
        out = rearrange(out, "b h n d -> b n (h d)")
    return out

def rotary_memory_efficient_attention(
    *,
    q, # (n, (h d), )
    k, # (n, (h d), )
    v, # (n, (h d), )
    rotary_time_emb_q, # (n, d)
    rotary_time_emb_kv, # (n, d)
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
    q_seqlen=None,
    kv_seqlen=None,
): # Output: (n, (h d), )
    r"""Wraps the memory efficient attention implementation with rotary embedding 
    application.
    """

    # xformers attention expects shape (1, n, h, d, ) <-- original code, not using xformers here.
    q = rearrange(q, "n (h d) -> n h d", h=num_heads).unsqueeze(0)
    k = rearrange(k, "n (h d) -> n h d", h=num_heads).unsqueeze(0)
    v = rearrange(v, "n (h d) -> n h d", h=num_heads).unsqueeze(0)

    q = apply_rotary_pos_emb(rotary_time_emb_q.unsqueeze(0), q)
    k = apply_rotary_pos_emb(rotary_time_emb_kv.unsqueeze(0), k)
    # if rotate_value:
    #     v = apply_rotary_pos_emb(rotary_time_emb_kv.unsqueeze(0), v)

    # Fill attention_bias with BlockDiagonalMask
    with torch.no_grad():
        # xformers expects 'list' of seqlens
        if q_seqlen is None:
            raise ValueError(
                f"`q_seqlen` must be a valid sequence length."
            )
        elif isinstance(q_seqlen, torch.Tensor):
            q_seqlen = q_seqlen.tolist()
        elif not isinstance(q_seqlen, list):
            raise ValueError(
                f"`q_seqlen` must be a list or a torch.Tensor, "
                f"got {type(q_seqlen)}"
            )

        if kv_seqlen is not None:
            # xformers expects 'list' of seqlens
            if isinstance(kv_seqlen, torch.Tensor):
                kv_seqlen = kv_seqlen.tolist()
            elif not isinstance(kv_seqlen, list):
                raise ValueError(
                    f"`kv_seqlen` must be a list or a torch.Tensor, "
                    f"got {type(kv_seqlen)}"
                )
            
        # attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
        #     q_seqlen=q_seqlen,
        #     kv_seqlen=kv_seqlen,
        # )
        attn_bias = None # Disable xformers (not supported on SM 12.0)

        # Following code changed to be compatible with 4500 Blackwell; disabling xformers (not supported on SM 12.0)
        # Reshape to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout if training else 0.0,
            is_causal=False
        )

        # Reshape back
        out = out.transpose(1, 2)