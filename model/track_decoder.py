"""
track_decoder.py
================
N_MAX persistent track queries, carried frame-to-frame via GRU hidden state.

Usage — call init_hidden() once per sequence, then forward_step() per frame:

    h = decoder.init_hidden(B, device)
    for t in range(T):
        states_t, logits_t, h = decoder.forward_step(enc_out_t, pad_mask_t, h)

Architecture per frame:
  1. Reshape h → queries  (B, N_MAX, d_model)
  2. n_layers × [cross-attn → self-attn → FFN]  (all pre-LayerNorm)
  3. GRUCell: h_next = GRU(attended_query, h_prev)   — integrates temporal memory
  4. Output heads on h_next: state (6D) and existence logit (1D)

The GRU hidden state *is* the track-query representation.  It is initialised
from a learned embedding so the model can learn a good prior for newborn tracks.
"""

import torch
import torch.nn as nn


class _FFN(nn.Module):
    """Pre-norm position-wise FFN with residual."""
    def __init__(self, d_model: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net  = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class TrackDecoder(nn.Module):

    def __init__(
        self,
        N_MAX    : int   = 5,
        d_model  : int   = 128,
        n_heads  : int   = 4,
        n_layers : int   = 2,
        dim_ff   : int   = 256,
        dropout  : float = 0.1,
    ):
        super().__init__()
        self.N_MAX   = N_MAX
        self.d_model = d_model

        # Learned initial query (one per track slot), shape (N_MAX, d_model).
        # Expanded to (B, N_MAX, d_model) at runtime and used as GRU h_0.
        self.query_init = nn.Parameter(torch.zeros(N_MAX, d_model))
        nn.init.normal_(self.query_init, std=0.02)

        # Per-layer modules ──────────────────────────────────────────────────
        self.cross_norm  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.self_norm   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.self_attns  = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffns = nn.ModuleList([_FFN(d_model, dim_ff, dropout) for _ in range(n_layers)])

        # GRU cell: input = attended query, hidden = track state ─────────────
        self.gru_cell = nn.GRUCell(d_model, d_model)

        # Output heads (applied to new hidden state) ─────────────────────────
        # state_head → 6D normalised Cartesian [x/300, y/300, z/300, vx/25, vy/25, vz/25]
        self.state_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 6),
        )
        # exist_head → scalar logit (apply sigmoid externally or use BCEWithLogitsLoss)
        self.exist_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    # ──────────────────────────────────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Return initial GRU hidden state shape (B*N_MAX, d_model).
        Expanded from the learned query_init parameter.
        """
        q0 = self.query_init.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N_MAX, d)
        return q0.reshape(batch_size * self.N_MAX, self.d_model).contiguous()

    def forward_step(
        self,
        enc_out  : torch.Tensor,   # (B, M, d_model) — encoder output
        pad_mask : torch.Tensor,   # (B, M)  bool — True = padded detection slot
        h_prev   : torch.Tensor,   # (B*N_MAX, d_model) — GRU hidden state
    ):
        """
        One frame of online tracking.
        Returns (pred_states, exist_logits, h_next):
          pred_states   : (B, N_MAX, 6)   normalised state estimates
          exist_logits  : (B, N_MAX)      raw existence logits
          h_next        : (B*N_MAX, d_model)
        """
        B = enc_out.shape[0]
        q = h_prev.reshape(B, self.N_MAX, self.d_model)   # (B, N_MAX, d)

        # n_layers × cross-attn + self-attn + FFN ────────────────────────────
        for cn, cross, sn, self_a, ffn in zip(
            self.cross_norm, self.cross_attns,
            self.self_norm,  self.self_attns,
            self.ffns,
        ):
            # Cross-attention: each track query attends to detection embeddings
            q_norm  = cn(q)
            q_cross, _ = cross(q_norm, enc_out, enc_out,
                               key_padding_mask=pad_mask, need_weights=False)
            q = q + q_cross

            # Self-attention: track queries interact with each other
            q_norm  = sn(q)
            q_self, _ = self_a(q_norm, q_norm, q_norm, need_weights=False)
            q = q + q_self

            # Position-wise FFN
            q = ffn(q)

        # GRU update — q carries the frame's attended evidence; h_prev carries
        # the track's temporal memory.  h_next fuses both.
        q_flat  = q.reshape(B * self.N_MAX, self.d_model)      # (B*N_MAX, d)
        h_next  = self.gru_cell(q_flat, h_prev)                # (B*N_MAX, d)
        q_out   = h_next.reshape(B, self.N_MAX, self.d_model)  # (B, N_MAX, d)

        # Output heads
        pred_states  = self.state_head(q_out)                  # (B, N_MAX, 6)
        exist_logits = self.exist_head(q_out).squeeze(-1)      # (B, N_MAX)

        return pred_states, exist_logits, h_next
