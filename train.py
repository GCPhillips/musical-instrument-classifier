import argparse
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

import mirdata
import torch.cuda
import seaborn as sn
import pandas as pd

from model import CNNClassifier, TransformerClassifier
from dataset import load_data
import matplotlib.pyplot as plt

from utils import save_model, load_model, INST_DICT


def train_model(tracks, splits, model, args: argparse.Namespace):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epochs = args.epochs
    lr = args.lr

    train_data = load_data(tracks, splits[0], use_mel=True)
    model = model.to(device)

    # adjust weights based on total number of occurrences in the dataset
    class_weights = 1 - (torch.tensor([1311, 1854, 1744, 3555, 6032, 477, 627, 5971]) / 21571).to(device)
    loss = torch.nn.NLLLoss(weight=class_weights)
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
            losses.append(l)
            optim.step()
        loss_avg = sum(losses) / len(losses)
        global_losses.append(loss_avg)
        print(f'epoch {epoch} - {loss_avg}')
    plt.plot(global_losses)
    plt.show()
    model.eval()
    return model


def test_model(model, tracks, splits):
    """
    Tests model against testing data from dataset
    :param model: the model instance to test
    :param tracks: the tracks loaded from the mirdata dataset
    :param splits: the train/test splits as a list
    """
    test_data = load_data(tracks, splits[1], training=False)
    preds = None
    targets = None

    for x, label in test_data:
        y_pred = model(x).cpu().detach()
        y_pred_cm = torch.argmax(y_pred, dim=1)
        label = label.cpu()
        if preds is None:
            preds = y_pred_cm
            targets = label
        else:
            preds = torch.cat((preds, y_pred_cm))
            targets = torch.cat((targets, label))
    print_report(preds, targets)


def print_report(preds, targets):
    """
    Prints the classification report and prints a heatmap of the confusion matrix
    :param preds: the tensor of predictions made from the model
    :param targets: the tensor of labels
    """
    cmtx = confusion_matrix(targets, preds)
    df = pd.DataFrame(cmtx, columns=INST_DICT, index=INST_DICT)
    heatmap = sn.heatmap(df, annot=True, fmt="d", cbar=False, xticklabels=INST_DICT, yticklabels=INST_DICT)
    plt.show()

    print(classification_report(targets, preds, target_names=INST_DICT))


def main(args: argparse.Namespace):
    dataset = mirdata.initialize('medley_solos_db', data_home='./data/medley_solos_db')

    # .gitkeep file should already be there. Download data if it doesn't exist.
    data_path_len = len(os.listdir('./data/medley_solos_db'))
    if data_path_len <= 1:
        dataset.download()
        dataset.validate()
    splits = dataset.get_random_track_splits([0.7, 0.3])
    model_name = args.model
    if model_name == 'cnn':
        model = CNNClassifier()
    elif model_name == 'transformer':
        model = TransformerClassifier()
    tracks = dataset.load_tracks()
    model = train_model(tracks, splits, model, args)
    now = datetime.now()
    filename = now.strftime("mi_classifier_%m-%d-%y_%H%M.th")
    save_model(model, filename)
    test_model(model, tracks, splits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="train",
        description="Training program for Medley Solo DB dataset",
    )
    parser.add_argument('--model', choices=['cnn', 'transformer'], default='cnn')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    main(parser.parse_args())
