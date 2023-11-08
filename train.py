import os
from datetime import datetime

import mirdata
import torch.cuda
from model import InstrumentClassifier
from dataset import load_data, SAMPLES
import matplotlib.pyplot as plt

from utils import save_model


def train_irmas(tracks, splits):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    d_input = SAMPLES
    d_model = 4410
    epochs = 50

    train_data = load_data(tracks, splits['train'])
    model = InstrumentClassifier(d_input=d_input, d_model=d_model, num_layers=3)
    model = model.to(device)
    loss = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    losses = []

    for epoch in range(epochs):
        model.train()
        for x, label in train_data:
            optim.zero_grad()
            y_pred = model(x)
            l = loss(y_pred, label)
            l.backward()
            losses.append(l.detach().item())
            optim.step()
        print(f'epoch {epoch} - {sum(losses) / len(losses)}')
    plt.plot(losses)
    plt.show()
    model.eval()
    return model


def test_irmas(model, tracks, splits):
    test_data = load_data(tracks, splits['test'])
    loss = torch.nn.BCEWithLogitsLoss()
    losses = []

    for x, label in test_data:
        y_pred = model(x)
        l = loss(y_pred, label)
        losses.append(l.detach().item())
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    irmas = mirdata.initialize('irmas', data_home='./data')

    # .gitkeep file should already be there. Download data if it doesn't exist.
    data_path_len = len(os.listdir('./data'))
    if data_path_len <= 1:
        irmas.download()
        irmas.validate()
    splits = irmas.get_track_splits()
    tracks = irmas.load_tracks()
    model = train_irmas(tracks, splits)
    now = datetime.now()
    # filename = now.strftime("mi_classifier_%m-%d-%y_%H%M.th")
    # save_model(model, filename)
    test_irmas(model, tracks, splits)
