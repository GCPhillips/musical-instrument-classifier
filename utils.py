import numpy as np

INST_DICT = [
    "clarinet",
    "distorted electric guitar",
    "female singer",
    "flute",
    "piano",
    "tenor saxophone",
    "trumpet",
    "violin"
]

IRMAS_INST_DICT = [
    "cel",
    "cla",
    "flu",
    "gac",
    "gel",
    "org",
    "pia",
    "sax",
    "tru",
    "vio",
    "voi",
]


def stereo_to_mono(audio: np.array) -> np.array:
    return np.mean(audio, axis=0)


def save_model(model, filename: str):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'trained_models', filename))


def load_model(model, filename: str):
    from torch import load
    from os import path
    model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'trained_models', filename), map_location='cpu'))
    return model
