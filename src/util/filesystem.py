import json
import os
import pickle
from typing import Any


def save_file(path: str, obj: Any, save_pickled=False):
    if os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    if save_pickled:
        with open(path, 'wb') as f:
            pickle.dump(obj, file=f, protocol=2)
        return

    if isinstance(obj, (dict, list)):
        print('Obj:', obj)
        with open(path, 'w') as f:
            f.write(json.dumps(obj))
        return

    with open(path, 'w') as f:
        f.write(str(obj))
