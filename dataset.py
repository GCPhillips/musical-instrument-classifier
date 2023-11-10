import random

import torch.cuda
import torchaudio.functional

from utils import INST_DICT
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SAMPLE_RATE = 20000
SECONDS = 2
SAMPLES = SAMPLE_RATE * SECONDS
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

class InstrumentDataset(Dataset):
    def __init__(self, irmas_tracks, track_ids, source_sample_rate, num_classes, device, training=True):
        self.tracks = track_ids
        self.irmas_tracks = irmas_tracks
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS).to(device)
        self.resample = torchaudio.transforms.Resample(source_sample_rate, SAMPLE_RATE).to(device)
        self.num_classes = num_classes
        self.device = device
        self.training = training

    def _transform_audio(self, track):
        audio = torch.from_numpy(track.audio[0]).to(device)
        sr = track.audio[1]
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            audio = self.resample(audio)
        rand_start = random.randint(0, SAMPLE_RATE)
        audio = audio[:,rand_start:].to(self.device)
        audio_len = audio.shape[1]
        if audio_len > SAMPLES:
            audio = audio[:, :SAMPLES]
        if audio_len < SAMPLES:
            offset = SAMPLES - audio_len
            padding = (0, offset)
            audio = torch.nn.functional.pad(audio, padding)
        audio = self.mel_spectrogram(audio)
        return audio

    def _create_label(self, instrument_list):
        label = torch.zeros(self.num_classes)
        for instrument in instrument_list:
            label[INST_DICT.index(instrument)] = 1
        return label

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, item):
        track = self.irmas_tracks[self.tracks[item]]
        audio = self._transform_audio(track)

        # use the predominant instrument if training.  Otherwise, use the list of instruments
        if self.training:
            label = self._create_label([track.predominant_instrument])
        else:
            label = self._create_label(track.instrument)
        return audio, label.to(self.device)


def load_data(irmas_tracks, track_ids, num_workers=0, batch_size=128, shuffle=True, drop_last=True, training=True):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = InstrumentDataset(irmas_tracks, track_ids, 44100, len(INST_DICT), device, training=training)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)