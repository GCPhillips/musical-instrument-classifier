import numpy as np
from model import InstrumentClassifier


INST_DICT = [
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


def save_model(model: InstrumentClassifier):
    from torch import save
    from os import path
    if isinstance(model, InstrumentClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'mi_classifier.th'))
    raise ValueError(f"model type '{str(type(model))}' is not supported!")


def load_model() -> InstrumentClassifier:
    from torch import load
    from os import path
    model = InstrumentClassifier()
    model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'mi_classifier.th'), map_location='cpu'))
    return model
