import inspect
from typing import Dict


def map_arguments(func, parameters: Dict) -> Dict:
    """ Extracts only theose parameters that are present in the function definition with their names as parameters """
    arguments = inspect.signature(func).parameters
    return {arg: parameters[arg] if arg in parameters else arguments[arg] for arg in arguments.keys()}
