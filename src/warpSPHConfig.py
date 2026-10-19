"""Lightweight configuration-file reading, safe before the precision bootstrap."""
from __future__ import annotations

import json
from pathlib import Path


def readMapping(path):
    path = Path(path)
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f'duplicate configuration key {key!r}')
            if not isinstance(key, str):
                raise ValueError('configuration keys must be strings')
            out[key] = value
        return out
    try:
        if path.suffix.lower() in ('.yaml', '.yml'):
            import yaml
            class UniqueLoader(yaml.SafeLoader):
                pass
            def construct(loader, node):
                # Reject duplicate explicit keys, while preserving YAML's defined
                # merge-key precedence (an explicit value may override an anchor).
                pairs((loader.construct_object(k), None) for k, _ in node.value
                      if k.tag != 'tag:yaml.org,2002:merge')
                loader.flatten_mapping(node)
                return {loader.construct_object(k): loader.construct_object(v)
                        for k, v in node.value}
            UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct)
            data = yaml.load(path.read_text(), Loader=UniqueLoader)
        else:
            data = json.loads(path.read_text(), object_pairs_hook=pairs)
    except ValueError as error:
        raise ValueError(f'{path}: {error}') from error
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected a top-level mapping')
    return data
