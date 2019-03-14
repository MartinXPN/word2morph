import json
from inspect import Parameter

import numpy as np
import os
import pickle
from typing import Any


def default_encoding(o):
    if isinstance(o, (np.int64, np.int32, np.int)):
        return int(o)
    if isinstance(o, (np.float32, np.float)):
        return float(o)
    if isinstance(o, Parameter):
        return o.name
    return str(o)


def save_file(path: str, obj: Any, save_pickled=False):
    if os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    if save_pickled:
        with open(path, 'wb') as f:
            pickle.dump(obj, file=f, protocol=2)
        return

    if isinstance(obj, (dict, list)):
        with open(path, 'w') as f:
            f.write(json.dumps(obj, default=default_encoding))
        return

    with open(path, 'w') as f:
        f.write(str(obj))
