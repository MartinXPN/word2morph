import inspect
import numpy as np
from typing import Dict


def map_arguments(func, parameters: Dict) -> Dict:
    """ Extracts only theose parameters that are present in the function definition with their names as parameters """
    arguments = {param.name: param.default for param in inspect.signature(func).parameters.values()}
    res = {arg: parameters[arg] if arg in parameters else arguments[arg] for arg in arguments.keys()}
    res = {key: value.item() if isinstance(value, (np.ndarray, np.generic)) else value for key, value in res.items()}
    return res
