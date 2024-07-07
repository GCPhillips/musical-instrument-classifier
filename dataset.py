import random

import torch.cuda
import torchaudio.functional
from transformers import ASTFeatureExtractor

from utils import INST_DICT, IRMAS_INST_DICT
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SAMPLE_RATE = 16000
SECONDS = 2
SAMPLES = SAMPLE_RATE * SECONDS
N_FFT = 1024
HOP_LENGTH = 128
N_MELS = 64


class MedleySolosDataset(Dataset):
    def __init__(self, tracks, track_ids, source_sample_rate, num_classes, device, use_mel=True):
        self.track_ids = track_ids
        self.tracks = tracks
        self.mel_spectrogram = ASTFeatureExtractor(num_mel_bins=N_MELS, sampling_rate=SAMPLE_RATE)
        self.resample = torchaudio.transforms.Resample(source_sample_rate, SAMPLE_RATE).to(device)
        self.num_classes = num_classes
        self.device = device
        self.use_mel = use_mel

    def _transform_audio(self, track):
        audio = torch.from_numpy(track.audio[0])
        audio = audio.to(device)
        audio = self.resample(audio)
        rand_start = random.randint(0, SAMPLE_RATE)
        audio = audio[rand_start:].to(self.device)
        audio_len = audio.shape[0]
        if audio_len > SAMPLES:
            audio = audio[:SAMPLES]
        if audio_len < SAMPLES:
            offset = SAMPLES - audio_len
            padding = (0, offset)
            audio = torch.nn.functional.pad(audio, padding)
        if self.use_mel:
            audio = self.mel_spectrogram(audio.cpu(), sampling_rate=SAMPLE_RATE, return_tensors='pt').data['input_values'].squeeze()
        return audio.to(device)

    def _create_label(self, instrument):
        return torch.tensor(INST_DICT.index(instrument))

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, item):
        track = self.tracks[self.track_ids[item]]
        audio = self._transform_audio(track)
        label = self._create_label(track.instrument)
        return audio, label.to(self.device)


class IrmasDataset(Dataset):
    def __init__(self, tracks, track_ids, source_sample_rate, num_classes, device, use_mel=True):
        self.track_ids = track_ids
        self.tracks = tracks
        self.mel_spectrogram = ASTFeatureExtractor(num_mel_bins=N_MELS, sampling_rate=SAMPLE_RATE)
        self.resample = torchaudio.transforms.Resample(source_sample_rate, SAMPLE_RATE).to(device)
        self.num_classes = num_classes
        self.device = device
        self.use_mel = use_mel

    def _transform_audio(self, track):
        audio = torch.from_numpy(track.audio[0])
        audio = audio.to(device)
        audio = self.resample(audio)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            audio = audio.squeeze()
        rand_start = random.randint(0, SAMPLE_RATE)
        audio = audio[rand_start:].to(self.device)
        audio_len = audio.shape[0]
        if audio_len > SAMPLES:
            audio = audio[:SAMPLES]
        if audio_len < SAMPLES:
            offset = SAMPLES - audio_len
            padding = (0, offset)
            audio = torch.nn.functional.pad(audio, padding)
        if self.use_mel:
            audio = self.mel_spectrogram(audio.cpu(), sampling_rate=SAMPLE_RATE, return_tensors='pt').data['input_values'].squeeze()
        return audio.to(device)

    def _create_label(self, instrument_list):
        label = torch.zeros(self.num_classes)
        for instrument in instrument_list:
            label[IRMAS_INST_DICT.index(instrument)] = 1.
        return label

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, item):
        track = self.tracks[self.track_ids[item]]
        audio = self._transform_audio(track)
        label = self._create_label(track.instrument)
        return audio, label.to(self.device)


def load_data(tracks, track_ids, num_workers=0, batch_size=128, shuffle=True, drop_last=True, use_mel=True):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = MedleySolosDataset(tracks, track_ids, 44100, len(INST_DICT), device, use_mel=use_mel)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def load_irmas_data(tracks, track_ids, num_workers=0, batch_size=128, shuffle=True, drop_last=True, use_mel=True):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = IrmasDataset(tracks, track_ids, 44100, len(IRMAS_INST_DICT), device, use_mel=use_mel)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
