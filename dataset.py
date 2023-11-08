import numpy as np
import torch.cuda

from utils import stereo_to_mono, INST_DICT
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SAMPLES = (44100 * 3) - 1  # 3 seconds at 44.1 khz, one less sample

class InstrumentDataset(Dataset):
    def __init__(self, irmas_tracks, track_ids):
        self.tracks = []
        self.labels = []

        for id in track_ids:
            track = irmas_tracks[id]
            audio = track.audio[0]
            if len(audio) == 2:
                audio = stereo_to_mono(audio)
            if audio.shape[0] > SAMPLES:
                audio = audio[0:SAMPLES]
            label = np.zeros(len(INST_DICT))
            for instr in track.instrument:
                label[INST_DICT.index(instr)] = 1.0
            label = torch.from_numpy(label)
            label = label.to(device)
            audio = torch.from_numpy(audio)
            audio = audio.to(device)
            self.tracks.append(audio)
            self.labels.append(label)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, item):
        return self.tracks[item], self.labels[item]


def load_data(irmas_tracks, track_ids, num_workers=0, batch_size=128, shuffle=True, drop_last=True):
    dataset = InstrumentDataset(irmas_tracks, track_ids)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)