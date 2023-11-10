import numpy as np
from model import InstrumentClassifier, CNN

INST_DICT = [
    "cel", # 388 examples
    "cla", # 505 examples
    "flu", # 451 examples
    "gac", # 637 examples
    "gel", # 760 examples
    "org", # 682 examples
    "pia", # 721 examples
    "sax", # 626 examples
    "tru", # 577 examples
    "vio", # 580 examples
    "voi", # 778 examples
]


def stereo_to_mono(audio: np.array) -> np.array:
    return np.mean(audio, axis=0)


def save_model(model: CNN, filename: str):
    from torch import save
    from os import path
    if isinstance(model, CNN):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'trained_models', filename))
    raise ValueError(f"model type '{str(type(model))}' is not supported!")


def load_model(filename: str) -> CNN:
    from torch import load
    from os import path
    model = CNN()
    model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'trained_models', filename), map_location='cpu'))
    return model
