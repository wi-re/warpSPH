"""`torch.utils.data.Dataset` wrapper over exported `.h5`/`.hdf5` trajectories.

`SPHDataset` scans a folder of exported runs and yields history/current/future
position-velocity-density slices plus static per-particle tensors and restored
run metadata per valid frame index; `sample_to_state`/`sample_to_domain`/
`SPHDataset.restore_state` rebuild a `WeaklyCompressibleState`/domain from a
sample for downstream use (e.g. resuming a simulation or seeding a surrogate
model). Used from the example notebooks, not from `runner`/`cases`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from ..configurations import dictToConfig
from .importIO import schemeNameToSimulationScheme
from .hdf5 import restore_config_from_h5
from ..schemes import buildScheme
from ..systems.weaklyCompressible import WeaklyCompressibleState


@dataclass
class DatasetParams:
    folder: str = field(default_factory=str)
    name: str = field(default_factory=str)
    history_length: int = 0
    trajectory_length: int = 0
    downsample_factor: int = 1
    cutoff_init: int = 0
    cutoff_final: int = 0


def sph_collate_variable(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep batch entries as a list for variable particle-count samples."""
    return batch


def sample_to_state(
    sample: dict[str, Any],
    remove_ghost: bool = True,
    device: str | torch.device = "cpu",
) -> WeaklyCompressibleState:
    """Build a WeaklyCompressibleState directly from a dataset sample dictionary."""
    static = sample["static"]
    pos = sample["input_current"]["positions"]
    vel = sample["input_current"]["velocities"]
    rho = sample["input_current"]["densities"]

    kinds_all = static["combinedKinds"].to(torch.int32)
    mask = kinds_all != 2 if remove_ghost else torch.ones_like(kinds_all, dtype=torch.bool)
    target_device = torch.device(device)

    return WeaklyCompressibleState(
        positions=pos[mask].to(torch.float32).to(target_device),
        supports=static["combinedSupports"][mask].to(torch.float32).to(target_device),
        masses=static["combinedMasses"][mask].to(torch.float32).to(target_device),
        densities=rho[mask].to(torch.float32).to(target_device),
        velocities=vel[mask].to(torch.float32).to(target_device),
        pressures=None,
        soundspeeds=None,
        kinds=static["combinedKinds"][mask].to(torch.int32).to(target_device),
        materials=static["combinedMaterials"][mask].to(torch.int32).to(target_device),
        emitters=(static["combinedEmitters"][mask].to(torch.int32).to(target_device)
                  if "combinedEmitters" in static else None),
        bodyIDs=(static["combinedBodyIDs"][mask].to(torch.int32).to(target_device)
                 if "combinedBodyIDs" in static else None),
        UIDs=static["combinedUIDs"][mask].to(torch.int64).to(target_device),
        UIDcounter=int(mask.sum().item()),
        ghostIndices=(
            static["combinedGhostIndices"][mask].to(torch.int32).to(target_device)
            if "combinedGhostIndices" in static
            else None
        ),
        ghostOffsets=(
            static["combinedGhostOffsets"][mask].to(torch.float32).to(target_device)
            if "combinedGhostOffsets" in static
            else None
        ),
    )


def sample_to_domain(
    sample: dict[str, Any],
    device: str | torch.device = "cpu",
):
    """Rebuild the domain description from a dataset sample's restored config metadata."""
    restored = sample["metadata"].get("restored_config")
    if restored is None:
        return None

    target_device = torch.device(device)
    cfg = dictToConfig(restored["config"])
    cfg.domain.min = cfg.domain.min.to(target_device)
    cfg.domain.max = cfg.domain.max.to(target_device)
    return cfg.domain


class SPHDataset(torch.utils.data.Dataset):
    """
    Dataset loader for compressed SPH trajectories (.h5/.hdf5).

    Returns one sample per valid frame index with history/current/future slices,
    static particle tensors, and metadata.
    """

    def __init__(self, params: DatasetParams):
        super().__init__()
        self.params = params
        self.folder = Path(params.folder)
        self.name = params.name if params.name else self.folder.name
        self.history_length = int(params.history_length)
        self.trajectory_length = int(params.trajectory_length)
        self.downsample_factor = max(1, int(params.downsample_factor))
        self.cutoff_init = max(0, int(params.cutoff_init))
        self.cutoff_final = max(0, int(params.cutoff_final))

        if not self.folder.exists():
            raise FileNotFoundError(f"Dataset folder does not exist: {self.folder}")

        self.simulation_files = sorted(
            [
                str(self.folder / fn)
                for fn in os.listdir(self.folder)
                if fn.endswith(".h5") or fn.endswith(".hdf5")
            ]
        )

        if len(self.simulation_files) == 0:
            raise RuntimeError(f"No .h5/.hdf5 files found in {self.folder}")

        # Keep behavior aligned with notebook usage: trajectory_length=0 means one future frame.
        self._future_length = max(1, self.trajectory_length)
        self._file_info: list[dict[str, Any]] = []
        self.valid_indices: list[tuple[int, int]] = []

        for file_idx, file_path in enumerate(self.simulation_files):
            info = self._scan_file(file_path)
            self._file_info.append(info)
            self.valid_indices.extend(
                [(file_idx, frame_idx) for frame_idx in info["valid_frame_indices"]]
            )

        if len(self.valid_indices) == 0:
            raise RuntimeError(
                "No valid dataset samples found. Check history/trajectory/downsample/cutoff settings."
            )

    def _scan_file(self, file_path: str) -> dict[str, Any]:
        with h5py.File(file_path, "r") as f:
            frame_keys: dict[str, list[str]] = {}
            for key in ["positions", "velocities", "densities", "times"]:
                if key in f and self._is_group(f, key):
                    keys = list(f[key].keys())
                    keys = sorted(keys, key=self._frame_key_sort_key)
                    frame_keys[key] = keys

            num_frames = self._num_frames(f, frame_keys)

            first_valid = self.cutoff_init + self.history_length * self.downsample_factor
            last_valid = (
                num_frames
                - 1
                - self.cutoff_final
                - self._future_length * self.downsample_factor
            )

            valid = [] if last_valid < first_valid else list(range(first_valid, last_valid + 1))

            return {
                "path": file_path,
                "num_frames": num_frames,
                "valid_frame_indices": valid,
                "frame_keys": frame_keys,
            }

    @staticmethod
    def _is_group(f: h5py.File, key: str) -> bool:
        return isinstance(f[key], h5py.Group)

    @staticmethod
    def _frame_key_sort_key(key: str) -> Any:
        if "_" in key:
            tail = key.split("_")[-1]
            if tail.isdigit():
                return int(tail)
        if key.isdigit():
            return int(key)
        return key

    def _num_frames(self, f: h5py.File, frame_keys: dict[str, list[str]]) -> int:
        if self._is_group(f, "positions"):
            return len(frame_keys.get("positions", []))
        return int(f["positions"].shape[0])

    def _read_frame_array(
        self,
        f: h5py.File,
        key: str,
        frame_idx: int,
        file_info: dict[str, Any],
    ) -> np.ndarray:
        if self._is_group(f, key):
            keys = file_info["frame_keys"].get(key, [])
            if frame_idx < 0 or frame_idx >= len(keys):
                raise IndexError(f"Frame index {frame_idx} out of range for key '{key}'")
            return f[key][keys[frame_idx]][:]
        return f[key][frame_idx][:]

    def _read_frame_time(
        self, f: h5py.File, frame_idx: int, file_info: dict[str, Any]
    ) -> float:
        if "times" not in f:
            return float("nan")
        if self._is_group(f, "times"):
            keys = file_info["frame_keys"].get("times", [])
            if frame_idx < 0 or frame_idx >= len(keys):
                return float("nan")
            return float(f["times"][keys[frame_idx]][:].item())
        return float(f["times"][frame_idx].item())

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _read_static(self, f: h5py.File) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key in [
            "combinedMasses",
            "combinedKinds",
            "combinedSupports",
            "combinedUIDs",
            "combinedMaterials",
            "combinedEmitters",
            "combinedBodyIDs",
            "combinedGhostIndices",
            "combinedGhostOffsets",
            "boundaryPositions",
            "boundaryMasses",
            "boundarySupports",
            "boundaryUIDs",
            "boundaryKinds",
            "boundaryGhostOffsets",
            "boundaryOffsets",
            "ghostPositions",
            "ghostMasses",
            "ghostSupports",
            "ghostUIDs",
            "ghostKinds",
        ]:
            if key in f:
                arr = f[key][:]
                if np.issubdtype(arr.dtype, np.integer):
                    out[key] = torch.as_tensor(arr, dtype=torch.int64)
                else:
                    out[key] = torch.as_tensor(arr, dtype=torch.float32)
        return out

    def _read_metadata(
        self,
        f: h5py.File,
        frame_idx: int,
        file_path: str,
        file_info: dict[str, Any],
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "h5_file": file_path,
            "frame_index": int(frame_idx),
            "time": self._read_frame_time(f, frame_idx, file_info),
            "dt_simulation": float(f.attrs["original_dt"])
            if "original_dt" in f.attrs
            else float("nan"),
            "dt_export": float(f.attrs["exportInterval"])
            if "exportInterval" in f.attrs
            else float("nan"),
            "num_particles": int(f["combinedPositions"].shape[0])
            if "combinedPositions" in f
            else int(self._read_frame_array(f, "positions", frame_idx, file_info).shape[0]),
            "num_fluid_particles": int((f["combinedKinds"][:] == 0).sum())
            if "combinedKinds" in f
            else -1,
            "num_boundary_particles": int((f["combinedKinds"][:] == 1).sum())
            if "combinedKinds" in f
            else -1,
        }

        attr_keys = [
            "caseName",
            "scheme",
            "targetNeighbors",
            "L",
            "W",
            "nx",
            "n_h",
            "timeLimit",
            "obstacleType",
            "obstacleActive",
            "aoa",
            "enableFreeSurface",
            "enableShifting",
            "gravityActive",
            "freeStreamVelocity",
            "enableFreeStream",
            "forcingWidth",
            "kolmogorovLengthScale",
            "enableKolmogorovForcing",
            "kolmogorovForcingAmplitude",
            "kolmogorovForcingWavenumber",
        ]
        for key in attr_keys:
            if key in f.attrs:
                value = f.attrs[key]
                if isinstance(value, np.ndarray):
                    meta[key] = value.tolist()
                elif isinstance(value, np.generic):
                    meta[key] = value.item()
                else:
                    meta[key] = value

        if "config" in f:
            try:
                meta["restored_config"] = restore_config_from_h5(f["config"])
            except Exception:
                meta["restored_config"] = None

        return meta

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx, frame_idx = self.valid_indices[idx]
        file_path = self.simulation_files[file_idx]
        file_info = self._file_info[file_idx]

        with h5py.File(file_path, "r") as f:
            history_frames = [
                frame_idx - (self.history_length - h) * self.downsample_factor
                for h in range(self.history_length)
            ]

            history_positions = []
            history_velocities = []
            history_densities = []
            for hf in history_frames:
                history_positions.append(self._read_frame_array(f, "positions", hf, file_info))
                history_velocities.append(self._read_frame_array(f, "velocities", hf, file_info))
                history_densities.append(self._read_frame_array(f, "densities", hf, file_info))

            current_positions = self._read_frame_array(f, "positions", frame_idx, file_info)
            current_velocities = self._read_frame_array(f, "velocities", frame_idx, file_info)
            current_densities = self._read_frame_array(f, "densities", frame_idx, file_info)

            trajectory_positions = []
            trajectory_velocities = []
            trajectory_densities = []
            for t in range(1, self._future_length + 1):
                tf = frame_idx + t * self.downsample_factor
                trajectory_positions.append(self._read_frame_array(f, "positions", tf, file_info))
                trajectory_velocities.append(self._read_frame_array(f, "velocities", tf, file_info))
                trajectory_densities.append(self._read_frame_array(f, "densities", tf, file_info))

            history_positions_t = (
                torch.as_tensor(np.stack(history_positions, axis=0), dtype=torch.float32)
                if len(history_positions) > 0
                else torch.empty((0,) + current_positions.shape, dtype=torch.float32)
            )
            history_velocities_t = (
                torch.as_tensor(np.stack(history_velocities, axis=0), dtype=torch.float32)
                if len(history_velocities) > 0
                else torch.empty((0,) + current_velocities.shape, dtype=torch.float32)
            )
            history_densities_t = (
                torch.as_tensor(np.stack(history_densities, axis=0), dtype=torch.float32)
                if len(history_densities) > 0
                else torch.empty((0,) + current_densities.shape, dtype=torch.float32)
            )

            sample = {
                "static": self._read_static(f),
                "metadata": self._read_metadata(f, frame_idx, file_path, file_info),
                "input_history": {
                    "positions": history_positions_t,
                    "velocities": history_velocities_t,
                    "densities": history_densities_t,
                },
                "input_current": {
                    "positions": torch.as_tensor(current_positions, dtype=torch.float32),
                    "velocities": torch.as_tensor(current_velocities, dtype=torch.float32),
                    "densities": torch.as_tensor(current_densities, dtype=torch.float32),
                },
                "target_trajectory": {
                    "positions": torch.as_tensor(
                        np.stack(trajectory_positions, axis=0), dtype=torch.float32
                    ),
                    "velocities": torch.as_tensor(
                        np.stack(trajectory_velocities, axis=0), dtype=torch.float32
                    ),
                    "densities": torch.as_tensor(
                        np.stack(trajectory_densities, axis=0), dtype=torch.float32
                    ),
                },
            }

            return sample

    def restore_state(
        self,
        sample_or_index: int | dict[str, Any],
        remove_ghost: bool = True,
        device: str | torch.device | None = None,
    ) -> tuple[WeaklyCompressibleState, Any]:
        if isinstance(sample_or_index, int):
            sample = self[sample_or_index]
        else:
            sample = sample_or_index

        meta = sample["metadata"]
        file_path = meta["h5_file"]
        frame_idx = int(meta["frame_index"])

        restored_config = meta.get("restored_config")
        if restored_config is None:
            raise RuntimeError("No restored config is available in the sample metadata.")

        scheme = restored_config["scheme"]
        scheme_enum = schemeNameToSimulationScheme(scheme)
        import_fn = buildScheme(scheme_enum).importFunction
        config = dictToConfig(restored_config["config"])
        _ = import_fn(restored_config["schemeConfig"])

        state_device = torch.device(config.device if device is None else device)

        file_idx, _ = (
            self.valid_indices[sample_or_index]
            if isinstance(sample_or_index, int)
            else (None, None)
        )
        if file_idx is None:
            file_idx = self.simulation_files.index(file_path)
        file_info = self._file_info[file_idx]

        with h5py.File(file_path, "r") as f:
            kinds_np = f["combinedKinds"][:]
            mask = kinds_np != 2 if remove_ghost else np.ones_like(kinds_np, dtype=bool)

            pos = self._read_frame_array(f, "positions", frame_idx, file_info)[mask]
            vel = self._read_frame_array(f, "velocities", frame_idx, file_info)[mask]
            rho = self._read_frame_array(f, "densities", frame_idx, file_info)[mask]

            state = WeaklyCompressibleState(
                positions=torch.as_tensor(pos, dtype=torch.float32, device=state_device),
                supports=torch.as_tensor(
                    f["combinedSupports"][:][mask], dtype=torch.float32, device=state_device
                ),
                masses=torch.as_tensor(
                    f["combinedMasses"][:][mask], dtype=torch.float32, device=state_device
                ),
                densities=torch.as_tensor(rho, dtype=torch.float32, device=state_device),
                velocities=torch.as_tensor(vel, dtype=torch.float32, device=state_device),
                pressures=None,
                soundspeeds=None,
                kinds=torch.as_tensor(
                    f["combinedKinds"][:][mask], dtype=torch.int32, device=state_device
                ),
                materials=torch.as_tensor(
                    f["combinedMaterials"][:][mask], dtype=torch.int32, device=state_device
                ),
                emitters=(torch.as_tensor(f["combinedEmitters"][:][mask], dtype=torch.int32, device=state_device)
                          if "combinedEmitters" in f else None),
                bodyIDs=(torch.as_tensor(f["combinedBodyIDs"][:][mask], dtype=torch.int32, device=state_device)
                         if "combinedBodyIDs" in f else None),
                UIDs=torch.as_tensor(
                    f["combinedUIDs"][:][mask], dtype=torch.int64, device=state_device
                ),
                UIDcounter=int(np.sum(mask)),
                ghostIndices=(
                    torch.as_tensor(
                        f["combinedGhostIndices"][:][mask],
                        dtype=torch.int32,
                        device=state_device,
                    )
                    if "combinedGhostIndices" in f
                    else None
                ),
                ghostOffsets=(
                    torch.as_tensor(
                        f["combinedGhostOffsets"][:][mask],
                        dtype=torch.float32,
                        device=state_device,
                    )
                    if "combinedGhostOffsets" in f
                    else None
                ),
            )

        return state, config


__all__ = [
    "DatasetParams",
    "SPHDataset",
    "sph_collate_variable",
    "sample_to_state",
    "sample_to_domain",
    "restore_config_from_h5",
]
