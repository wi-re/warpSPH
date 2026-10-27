"""Parameter metadata shared by configuration validation, documentation and clients.

Defaults remain on Case.params / CaseSpec / the solver configuration. Descriptors
supply semantics rather than another independent set of default values.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class Parameter:
    kind: str
    description: str
    group: str = 'case'
    units: Optional[str] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    exclusiveMinimum: Optional[float] = None
    exclusiveMaximum: Optional[float] = None
    choices: Tuple[Any, ...] = ()
    nullable: bool = False
    schemes: Tuple[str, ...] = ()
    appliesWhen: Optional[dict] = None
    appliesWhenAny: Tuple[dict, ...] = ()
    items: Optional['Parameter'] = None
    properties: Optional[Dict[str, 'Parameter']] = None
    required: Tuple[str, ...] = ()
    minItems: Optional[int] = None
    maxItems: Optional[int] = None
    readOnly: bool = False
    derivation: Optional[str] = None
    deprecated: bool = False

    def validate(self, value, path):
        if value is None and self.nullable:
            return
        valid = {
            'number': type(value) in (int, float),
            'integer': type(value) is int,
            'boolean': type(value) is bool,
            'string': isinstance(value, str),
            'object': isinstance(value, dict),
            'array': isinstance(value, list),
        }.get(self.kind, False)
        if not valid:
            raise ValueError(f'{path}: expected {self.kind}' + (' or null' if self.nullable else ''))
        if self.kind in ('number', 'integer'):
            if not math.isfinite(value):
                raise ValueError(f'{path}: must be finite')
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f'{path}: must be >= {self.minimum}')
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f'{path}: must be <= {self.maximum}')
            if self.exclusiveMinimum is not None and value <= self.exclusiveMinimum:
                raise ValueError(f'{path}: must be > {self.exclusiveMinimum}')
            if self.exclusiveMaximum is not None and value >= self.exclusiveMaximum:
                raise ValueError(f'{path}: must be < {self.exclusiveMaximum}')
        if self.kind == 'array':
            if self.minItems is not None and len(value) < self.minItems:
                raise ValueError(f'{path}: needs at least {self.minItems} items')
            if self.maxItems is not None and len(value) > self.maxItems:
                raise ValueError(f'{path}: needs at most {self.maxItems} items')
            if self.items:
                for i, item in enumerate(value):
                    self.items.validate(item, f'{path}[{i}]')
        if self.kind == 'object' and self.properties is not None:
            if any(not isinstance(k, str) for k in value):
                raise ValueError(f'{path}: keys must be strings')
            unknown = set(value) - set(self.properties)
            missing = set(self.required) - set(value)
            if unknown or missing:
                raise ValueError(f'{path}: unknown fields {sorted(unknown)}, missing fields {sorted(missing)}')
            for name, item in value.items():
                self.properties[name].validate(item, f'{path}.{name}')
        if self.choices and value not in self.choices:
            raise ValueError(f'{path}: expected one of {list(self.choices)}')

    def schema(self, default=...):
        out = {'type': [self.kind, 'null'] if self.nullable else self.kind,
               'description': self.description, 'x-group': self.group}
        if default is not ...:
            out['default'] = default
        for key in ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'minItems', 'maxItems'):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.choices:
            out['enum'] = list(self.choices) + ([None] if self.nullable else [])
        if self.items:
            out['items'] = self.items.schema()
        if self.properties is not None:
            out['properties'] = {k: p.schema() for k, p in self.properties.items()}
            out['additionalProperties'] = False
            if self.required:
                out['required'] = list(self.required)
        if self.readOnly:
            out['readOnly'] = True
        if self.derivation:
            out['x-derived-from'] = self.derivation
        if self.deprecated:
            out['deprecated'] = True
        if self.units:
            out['x-units'] = self.units
        if self.schemes:
            out['x-supported-schemes'] = list(self.schemes)
        if self.appliesWhenAny:
            out['x-applies-when-any'] = list(self.appliesWhenAny)
        if self.appliesWhen:
            out['x-applies-when'] = self.appliesWhen
        return out
