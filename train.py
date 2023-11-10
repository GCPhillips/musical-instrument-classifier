import os
from datetime import datetime

import mirdata
import torch.cuda

from model import CNN
from dataset import load_data
import matplotlib.pyplot as plt

from utils import save_model, load_model, INST_DICT


def train_irmas(tracks, splits):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epochs = 150
    lr = 1e-4

    train_data = load_data(tracks, splits['train'])
    model = CNN(n_output=11).to(device)
    # model = load_model('mi_classifier_11-10-23_0703.th').to(device)
    class_weights = 1 - (torch.tensor([388, 505, 451, 637, 760, 682, 721, 626, 577, 580, 778]) / 6705).to(device)
    loss = torch.nn.BCEWithLogitsLoss(weight=class_weights)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    global_losses = []
    for epoch in range(epochs):
        losses = []
        model.train()
        for x, label in train_data:
            optim.zero_grad()
            y_pred = model(x)
            l = loss(y_pred, label)
            l.backward()
            l = l.detach().item()
            if l <= 5:
                losses.append(l)
            optim.step()
        loss_avg = sum(losses) / len(losses)
        global_losses.append(loss_avg)
        print(f'epoch {epoch} - {loss_avg}')
    plt.plot(global_losses)
    plt.show()
    model.eval()
    return model


def test_irmas(model, tracks, splits):
    test_data = load_data(tracks, splits['test'], training=False)

    for x, label in test_data:
        y_pred = torch.sigmoid(model(x))
        y_pred_ex = (y_pred[1] > 0.2).nonzero(as_tuple=False)
        insts = ""
        for idx in y_pred_ex:
            insts += " " + INST_DICT[idx]
        labels = ""
        label_ex = label[1]
        for idx in (label_ex > 0).nonzero(as_tuple=False):
            labels += " " + INST_DICT[idx]
        print("Prediction: " + insts + "\nLabels:     " + labels + "\n")


def main():
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
    filename = now.strftime("mi_classifier_%m-%d-%y_%H%M.th")
    save_model(model, filename)
    # test_irmas(model, tracks, splits)


if __name__ == '__main__':
    # main()
    model = load_model('mi_classifier_11-10-23_1617.th').to(torch.device('cuda'))
    irmas = mirdata.initialize('irmas', data_home='./data')
    splits = irmas.get_track_splits()
    tracks = irmas.load_tracks()
    test_irmas(model, tracks, splits)
