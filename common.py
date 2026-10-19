"""Shared validation, period indexing and JSON output for the fixed focal panel."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

CREATORS = (
    'albert_cancook', 'chefadamlibby', 'doobydobap', 'eitan',
    'jacksdiningroom', 'kennylsong', 'notorious_foodie', 'the_pastaqueen',
    'thegoldenbalance', 'wishbonekitchen',
)
BLOCKS = ('conceptual', 'audiovisual', 'embodied')

def check_creators(values, exact=True):
    actual = set(values)
    if not actual <= set(CREATORS) or (exact and actual != set(CREATORS)):
        raise ValueError('Input must contain the fixed focal panel, not a candidate pool.')

def require_columns(frame, names):
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

def period_index(value):
    year, half = str(value).split('H')
    if half not in ('1','2'):
        raise ValueError(f'Invalid half-year period: {value}')
    return int(year) * 2 + int(half) - 1

def next_period(value, step=1):
    index = period_index(value) + step
    return f'{index // 2}H{index % 2 + 1}'

def stable_seed(*parts, base=1729):
    raw = '|'.join(map(str, parts)).encode('utf-8')
    return (base + int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little')) % (2**32 - 1)

def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')

def read_frame(path):
    return pd.read_csv(path, dtype={'content_id':str, 'event_content_id':str})

