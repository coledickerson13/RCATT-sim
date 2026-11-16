"""
agnostic_transformer.py
=======================
Online frame-by-frame multi-target tracker: wraps DetectionEncoder +
TrackDecoder and loops over T frames while propagating GRU state.

Ablation variants — all share this class, controlled by constructor flags:

  Tag   use_quality  use_context  Description
  ───   ───────────  ───────────  ────────────────────────────────────────────
  T1    False        False        Agnostic  (no quality, no resource context)
  T2    True         False        Quality-aware
  T3    False        True         RCATT     (resource context, no quality)  ←future
  T4    True         True         RCATT + quality                           ←future

For T3/T4 the `contexts` input (B, T, 5) will be injected into the encoder
output via a learned projection + additive conditioning.  The decoder and
loss are identical across all variants, ensuring a fair ablation.

State representation:
  Normalised Cartesian  [x/300, y/300, z/300, vx/25, vy/25, vz/25].
  Call norm_states(gt)  to bring GT into this space before loss computation.
  Call denorm_states(pred) to recover physical units (m, m/s).
"""

import torch
import torch.nn as nn

from .detection_encoder import DetectionEncoder
from .track_decoder     import TrackDecoder


# ── State normalisation constants ─────────────────────────────────────────────
# [x, y, z, vx, vy, vz] in metres / m·s⁻¹
# POS_SCALE = SURV_RHO_MAX = 300 m;  VEL_SCALE = V_MAX_MPS = 25 m/s
_POS_SCALE = 300.0
_VEL_SCALE  = 25.0
_STATE_SCALE = torch.tensor(
    [_POS_SCALE, _POS_SCALE, _POS_SCALE,
     _VEL_SCALE, _VEL_SCALE, _VEL_SCALE],
    dtype=torch.float32,
)

# ── Context normalisation constants ───────────────────────────────────────────
# [B_PRS_MHz, N_pil, T_obs_ms, P_tx_dBm, P_SI_dBm]
# Rough scales from sim_config (see RESOURCE_CONFIGS).  Used only when T3/T4.
_CTX_SCALE = torch.tensor(
    [100.0,    # B_PRS_MHz:  range ≈ 5 – 100 MHz
     60000.0,  # N_pil:      range ≈ 4 608 – 52 224
     10.0,     # T_obs_ms:   range ≈ 2 – 8 ms
     30.0,     # P_tx_dBm:   range ≈ 20 – 30 dBm
     -60.0,    # P_SI_dBm:   residual SI power (negative dBm); use sign-flip trick
     ],
    dtype=torch.float32,
)


class AgnosticTransformer(nn.Module):
    """
    Online multi-target tracker — processes one frame at a time and propagates
    track-query GRU state across the full T-frame sequence.

    Constructor args
    ----------------
    N_MAX        : int    max simultaneous targets (= N_TARGET_MAX from sim_config)
    d_model      : int    hidden dimension
    n_heads      : int    attention heads (must divide d_model)
    n_enc_layers : int    Transformer encoder depth
    n_dec_layers : int    Transformer decoder depth
    dim_ff       : int    FFN inner dimension
    dropout      : float  dropout probability
    use_quality  : bool   include Q_dB (det channel 7) in encoder input
    use_context  : bool   inject 5D resource context (RCATT; False → T1/T2)
    """

    def __init__(
        self,
        N_MAX        : int   = 5,
        d_model      : int   = 128,
        n_heads      : int   = 4,
        n_enc_layers : int   = 2,
        n_dec_layers : int   = 2,
        dim_ff       : int   = 256,
        dropout      : float = 0.1,
        use_quality  : bool  = False,   # T1 default
        use_context  : bool  = False,   # Agnostic default
    ):
        super().__init__()
        self.N_MAX       = N_MAX
        self.use_quality = use_quality
        self.use_context = use_context

        # State normalisation buffer (auto moves to device)
        self.register_buffer("state_scale", _STATE_SCALE)

        # Encoder — embeds + self-attends over detections at each frame
        self.encoder = DetectionEncoder(
            d_model     = d_model,
            n_heads     = n_heads,
            n_layers    = n_enc_layers,
            dim_ff      = dim_ff,
            dropout     = dropout,
            use_quality = use_quality,
        )

        # Decoder — persistent GRU track queries + cross-attention
        self.decoder = TrackDecoder(
            N_MAX    = N_MAX,
            d_model  = d_model,
            n_heads  = n_heads,
            n_layers = n_dec_layers,
            dim_ff   = dim_ff,
            dropout  = dropout,
        )

        # Context conditioning (T3/T4 only) ───────────────────────────────────
        # Projects 5D resource context to d_model and adds to enc_out additively.
        # For T1/T2 this module is instantiated but never called (zero overhead).
        if use_context:
            self.register_buffer("ctx_scale", _CTX_SCALE.abs())  # abs for P_SI sign
            self.ctx_proj = nn.Sequential(
                nn.Linear(5, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Normalisation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def norm_states(self, states: torch.Tensor) -> torch.Tensor:
        """Scale GT states to match model output range (÷ [300,300,300,25,25,25])."""
        return states / self.state_scale

    def denorm_states(self, pred: torch.Tensor) -> torch.Tensor:
        """Convert normalised model output back to physical units (m, m/s)."""
        return pred * self.state_scale

    # ──────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        detections : torch.Tensor,             # (B, T, M, 8)
        det_valid  : torch.Tensor,             # (B, T, M)   bool
        contexts   : torch.Tensor | None = None,  # (B, T, 5)  float; required if use_context
    ):
        """
        Returns
        -------
        pred_states       : (B, T, N_MAX, 6)   normalised Cartesian estimates
        pred_exist_logits : (B, T, N_MAX)       raw existence logits (pre-sigmoid)
        """
        B, T, M, _ = detections.shape
        device = detections.device

        # T3/T4 guard: catch missing context early so the error is actionable
        if self.use_context and contexts is None:
            raise ValueError(
                "AgnosticTransformer: contexts=None but use_context=True (T3/T4). "
                "Pass a (B, T, 5) context tensor."
            )

        pred_states_list  = []
        pred_logits_list  = []

        h = self.decoder.init_hidden(B, device)   # (B*N_MAX, d_model)

        for t in range(T):
            dets_t  = detections[:, t]    # (B, M, 8)
            valid_t = det_valid[:, t]     # (B, M)

            enc_out, pad_mask = self.encoder(dets_t, valid_t)   # (B, M, d_model)

            # Inject resource context for RCATT variants (T3/T4)
            if self.use_context:
                ctx_t = contexts[:, t] / self.ctx_scale          # (B, 5) normalised
                ctx_emb = self.ctx_proj(ctx_t).unsqueeze(1)      # (B, 1, d_model)
                enc_out = enc_out + ctx_emb                       # broadcast over M

            states_t, logits_t, h = self.decoder.forward_step(enc_out, pad_mask, h)

            pred_states_list.append(states_t)
            pred_logits_list.append(logits_t)

        pred_states       = torch.stack(pred_states_list,  dim=1)   # (B, T, N_MAX, 6)
        pred_exist_logits = torch.stack(pred_logits_list,  dim=1)   # (B, T, N_MAX)

        return pred_states, pred_exist_logits
