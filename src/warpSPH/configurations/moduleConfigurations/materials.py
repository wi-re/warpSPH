"""Serializable pure-material EOS descriptions. List position is the fluid material ID."""
from dataclasses import asdict, dataclass, field
import math

__all__ = ['EOSConfig', 'MaterialConfig', 'normalizeMaterials', 'materialsToDict']


@dataclass(frozen=True)
class EOSConfig:
    type: str = 'idealGas'
    gamma: float = 1.4

    def __post_init__(self):
        if self.type != 'idealGas':
            raise ValueError(f'materials.eos.type: unsupported EOS {self.type!r}')
        if type(self.gamma) not in (float, int) or not math.isfinite(self.gamma) or self.gamma <= 1:
            raise ValueError('materials.eos.gamma: must be finite and greater than 1')


@dataclass(frozen=True)
class MaterialConfig:
    name: str = 'gas'
    eos: EOSConfig = field(default_factory=EOSConfig)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('materials.name: expected a nonempty string')
        if isinstance(self.eos, dict):
            object.__setattr__(self, 'eos', EOSConfig(**self.eos))
        if not isinstance(self.eos, EOSConfig):
            raise ValueError('materials.eos: expected an EOSConfig or mapping')


def normalizeMaterials(materials):
    if materials is None:
        return None
    if not isinstance(materials, (list, tuple)) or not materials:
        raise ValueError('materials: expected a nonempty material list or None')
    result = []
    for i, entry in enumerate(materials):
        try:
            result.append(entry if isinstance(entry, MaterialConfig) else MaterialConfig(**entry))
        except (TypeError, ValueError) as exc:
            raise ValueError(f'materials[{i}]: {exc}') from exc
    return result


def materialsToDict(materials):
    return None if materials is None else [asdict(m) for m in normalizeMaterials(materials)]
