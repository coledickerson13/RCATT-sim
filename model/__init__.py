"""
model — RCATT tracker model package.

Ablation variants (all share the same backbone):
  T1  AgnosticTransformer(use_quality=False, use_context=False)  ← start here
  T2  AgnosticTransformer(use_quality=True,  use_context=False)
  T3  RCATT              (use_quality=False, use_context=True)   ← future
  T4  RCATT+Q            (use_quality=True,  use_context=True)   ← future
"""
from .agnostic_transformer import AgnosticTransformer
from .losses import compute_loss
from .dataloader import RCATTDataset, make_dataloader

__all__ = ["AgnosticTransformer", "compute_loss", "RCATTDataset", "make_dataloader"]
