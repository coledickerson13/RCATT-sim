"""
detection_encoder.py
====================
Embeds the per-frame detection set and runs a Transformer encoder over it.

Input  : dets      (B, M, 8)  raw detection features
                               [R_m, vr_mps, az_deg, el_deg,
                                beam_az, beam_el, dt_s, Q_dB]
         det_valid (B, M)     bool — True = real detection
Flags  : use_quality — if False, channel 7 (Q_dB) is zeroed before MLP
Output : enc_out   (B, M, d_model)   encoder output (padded slots included)
         pad_mask  (B, M)            True = padded slot (key_padding_mask)

Normalisation constants are derived directly from sim_config limits so the
model never sees raw metres or degrees.  They are registered as buffers so
they move to GPU automatically.

Edge-case guard: if every detection slot in a batch row is padded, the
Transformer would receive all-masked keys → softmax over −∞ → NaN.
We unmask all slots for such rows (all-zero embeddings are harmless).
"""

import torch
import torch.nn as nn


# ── Fixed input normalisation (one scalar per detection feature channel) ──────
# Derived from sim_config: SURV_RHO_MAX=300, V_MAX_MPS=25, AZ±60, EL 10–40,
# T_FRAME=0.5.  Q_dB: pilot diagnostics show [5, 45] dB range; use 50 as scale.
_DET_SCALE = torch.tensor([
    300.0,  # R_m        (0 – 300 m)
     25.0,  # vr_mps     (± 25 m/s)
     60.0,  # az_deg     (± 60°)
     40.0,  # el_deg     (10 – 40°; model learns the offset)
     60.0,  # beam_az    (± 60°)
     40.0,  # beam_el    (10 – 40°)
      0.5,  # dt_s       (0 – T_FRAME = 0.5 s)
     50.0,  # Q_dB       (0 – ~50 dB)
], dtype=torch.float32)


class DetectionEncoder(nn.Module):
    """
    Two-layer MLP input projection followed by a Transformer encoder with
    pre-LayerNorm (more stable gradients than post-norm in early training).
    """

    def __init__(
        self,
        d_model     : int   = 128,
        n_heads     : int   = 4,
        n_layers    : int   = 2,
        dim_ff      : int   = 256,
        dropout     : float = 0.1,
        use_quality : bool  = True,
    ):
        super().__init__()
        self.use_quality = use_quality
        self.register_buffer("det_scale", _DET_SCALE)

        # Input MLP: 8 → d_model (Q channel zeroed before this for agnostic models)
        self.input_proj = nn.Sequential(
            nn.Linear(8, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer encoder (pre-norm, batch_first)
        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward = dim_ff,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # pre-LayerNorm
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        dets      : torch.Tensor,   # (B, M, 8)
        det_valid : torch.Tensor,   # (B, M)  bool — True = real detection
    ):
        B, M, _ = dets.shape

        # 1. Normalise raw features
        x = dets / self.det_scale.view(1, 1, 8)   # (B, M, 8)

        # 2. Zero quality channel for quality-agnostic ablations (T1, T3)
        if not self.use_quality:
            x = x.clone()
            x[..., 7] = 0.0

        # 3. Project to d_model
        x = self.input_proj(x)   # (B, M, d_model)

        # 4. Build padding mask: True = padded (PyTorch MHA convention)
        pad_mask = ~det_valid   # (B, M)

        # 5. Guard against all-padded rows (would produce NaN in softmax).
        #    For rows where every slot is masked, unmask all — the all-zero
        #    embeddings are benign and the cross-attention will simply spread
        #    uniformly over them.
        all_padded = pad_mask.all(dim=1, keepdim=True)        # (B, 1)
        safe_mask  = pad_mask & ~all_padded.expand(B, M)      # (B, M)

        # 6. Transformer encoder (self-attention over detection set)
        x = self.encoder(x, src_key_padding_mask=safe_mask)   # (B, M, d_model)

        return x, safe_mask
