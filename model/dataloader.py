"""
dataloader.py
=============
HDF5 dataset + DataLoader for RCATT training.

HDF5 layout (from generate_dataset.py):
  contexts    (N_seq, T, 5)         float32   [B_PRS_MHz, N_pil, T_obs_ms, P_tx_dBm, P_SI_dBm]
  detections  (N_seq, T, M_max, 8)  float32
  det_valid   (N_seq, T, M_max)     bool
  labels      (N_seq, T, M_max)     int32     0=clutter, >0=target_id, -1=pad
  states      (N_seq, T, N_max, 6)  float32   ground-truth [x,y,z,vx,vy,vz]
  state_valid (N_seq, T, N_max)     bool
  visible     (N_seq, T, N_max)     bool
  state_ids   (N_seq, T, N_max)     int32     persistent target ID; −1 = padded
  cfg_keys    (N_seq, T)            uint8
  sic_dbs     (N_seq, T)            float32

Each __getitem__ returns a dict of Tensors with shapes as above but without the
leading N_seq dimension.

Usage
-----
  from model.dataloader import make_dataloader
  loader = make_dataloader("data/train_pilot.h5", batch_size=8, shuffle=True)
  for batch in loader:
      detections   = batch['detections']    # (B, T, M, 8)
      det_valid    = batch['det_valid']     # (B, T, M) bool
      states       = batch['states']        # (B, T, N_MAX, 6)
      state_valid  = batch['state_valid']   # (B, T, N_MAX) bool
      visible      = batch['visible']       # (B, T, N_MAX) bool
      contexts     = batch['contexts']      # (B, T, 5)
      target_mask  = state_valid & visible  # supervision mask
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RCATTDataset(Dataset):
    """
    Lazy-loading HDF5 dataset.  Each HDF5 file is opened once per worker in
    __getitem__ (lazy open pattern avoids h5py multiprocessing issues).

    Parameters
    ----------
    h5_path  : str   path to train_pilot.h5 / train.h5 / val.h5 etc.
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._file   = None   # opened lazily in __getitem__

        # Read length from file without keeping it open
        with h5py.File(h5_path, 'r') as f:
            self._len = f['states'].shape[0]

    def __len__(self) -> int:
        return self._len

    def _get_file(self) -> h5py.File:
        """Open the HDF5 file for this worker (once per worker process)."""
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r', swmr=True)
        return self._file

    def __getitem__(self, idx: int) -> dict:
        f = self._get_file()
        return {
            'detections'  : torch.from_numpy(f['detections' ][idx].astype(np.float32)),
            'det_valid'   : torch.from_numpy(f['det_valid'  ][idx].astype(bool)),
            'states'      : torch.from_numpy(f['states'     ][idx].astype(np.float32)),
            'state_valid' : torch.from_numpy(f['state_valid'][idx].astype(bool)),
            'visible'     : torch.from_numpy(f['visible'    ][idx].astype(bool)),
            'contexts'    : torch.from_numpy(f['contexts'   ][idx].astype(np.float32)),
            'state_ids'   : torch.from_numpy(f['state_ids'  ][idx].astype(np.int32)),
            'cfg_keys'    : torch.from_numpy(f['cfg_keys'   ][idx].astype(np.int64)),
            'sic_dbs'     : torch.from_numpy(f['sic_dbs'    ][idx].astype(np.float32)),
        }

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


def make_dataloader(
    h5_path    : str,
    batch_size : int  = 8,
    shuffle    : bool = True,
    num_workers: int  = 0,
    pin_memory : bool = True,
) -> DataLoader:
    """
    Construct a DataLoader over an RCATT HDF5 file.

    Parameters
    ----------
    h5_path     : path to .h5 file produced by generate_dataset.py
    batch_size  : sequences per batch
    shuffle     : True for train; False for val/test
    num_workers : number of background loading workers; 0 = main process only
                  (use 0 when debugging or on systems where fork is unreliable)
    pin_memory  : pin host memory for faster GPU transfers

    Returns
    -------
    DataLoader whose batches are dicts of Tensors with a leading batch dimension.
    """
    ds = RCATTDataset(h5_path)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = pin_memory and torch.cuda.is_available(),
        drop_last   = False,
        persistent_workers = (num_workers > 0),
    )
