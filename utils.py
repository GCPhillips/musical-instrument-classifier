import numpy as np
from model import InstrumentClassifier, CNN

INST_DICT = [
    "cel", #388
    "cla", #505
    "flu", #451
    "gac", #637
    "gel", #760
    "org", #682
    "pia", #721
    "sax", #626
    "tru", #577
    "vio", #580
    "voi", #778
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
