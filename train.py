import os
import random

import mirdata
import torch.cuda
from model import InstrumentClassifier
from dataset import load_data


def train_irmas(dataset):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seq_length = 44100
    random_samples = 100
    epochs = 15

    splits = dataset.get_track_splits()
    tracks = dataset.load_tracks()
    train_data = load_data(tracks, splits['train'])
    model = InstrumentClassifier(seq_length=seq_length)
    model = model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):

        losses = []
        for x, label in train_data:
            for iter in range(random_samples):
            # take a random sample
                i = random.randint(0, len(x[-1]) - seq_length - 1)
                x_slice = x[:,i:i+seq_length]
                x_slice = x_slice.to(device)
                y_pred = model(x_slice)
                l = loss(y_pred, label)
                l.backward()
                losses.append(l.detach().item())
                optim.step()
        print(f'epoch {epoch} - {sum(losses) / len(losses)}')


if __name__ == '__main__':
    irmas = mirdata.initialize('irmas', data_home='./data')

    # .gitkeep file should already be there. Download data if it doesn't exist.
    data_path_len = len(os.listdir('./data'))
    if data_path_len <= 1:
        irmas.download()
        irmas.validate()
    train_irmas(irmas)