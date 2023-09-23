"""
Import below any model you wanna use. Then, register the model by using register_model()
"""

import inspect
import os
import importlib.util
from policy.torch_model import TorchModel

# example
from policy.torch_model import LinearPPOModel

# dict for saving model classes
model_catalog = {

}

def register_model(model_cls_name, key):
    """
    Register a torch model
    :param model_cls_name: class name for the model
    :param key: unique key used ad id for the model class
    :return: None
    """
    path_python_module = inspect.getfile(model_cls_name)
    model_catalog[key] = {}
    model_catalog[key]["name"] = model_cls_name
    model_catalog[key]["path"] = path_python_module


def import_model(key, config):
    """
    This function returns an instance of the model class associated to the key.
    :param key: key to the model class
    :param config: config dict for the model
    :return: instance of TorchModel or subclass
    """
    path = model_catalog[key]["path"]
    model_cls_name = model_catalog[key]["name"]
    spec = importlib.util.spec_from_file_location("torch_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = getattr(module, model_cls_name.__name__)(config)

    return instance


"""Call register_model to register any torch model class you need"""

register_model(LinearPPOModel, "example")