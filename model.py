from transformers import ASTForAudioClassification, ASTConfig

import torch
import torch.nn as nn

from dataset import N_MELS

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class CNNClassifier(nn.Module):
    def __init__(self, n_output=8, use_logsoftmax=True):
        """
        Convolutional Neural Network with 3 layers

        :param n_output: number of classes to output
        """
        super().__init__()
        self.conv1 = CNNBlock(1, 32, 3, 2, 2)
        self.conv2 = CNNBlock(32, 64, 3, 2, 2)
        self.conv3 = CNNBlock(64, 128, 3, 2, 2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(148608, n_output)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.use_logsoftmax = use_logsoftmax

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear(x)
        if self.use_logsoftmax:
            x = self.logsoftmax(x)

        return x


class CNNBlock(nn.Module):
    def __init__(self, n_input, n_output, conv_kernel_size, pool_kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=n_input,
                              out_channels=n_output,
                              kernel_size=conv_kernel_size,
                              padding=padding)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class TransformerClassifier(nn.Module):
    def __init__(self, num_classes=8, num_layers=2, num_attention_heads=12, use_logsoftmax=True):
        """
        Transformer classifier using HuggingFace ASTForAudioClassification

        :param d_input: the dimensions of the classifier input
        :param d_model: the dimensions of the input and output of the transformer
        :param d_internal: the dimensions of the hidden layer in self-attention
        :param num_classes: the number of classes to predict (i.e. the number of musical instruments to classify)
        :param num_layers: the number of transformer layers to use
        """
        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        config = ASTConfig(num_mel_bins=N_MELS, num_attention_heads=num_attention_heads, num_hidden_layers=num_layers, num_labels=num_classes)
        self.ast = ASTForAudioClassification(config)
        self.use_logsoftmax = use_logsoftmax

    def forward(self, x):
        x = self.ast(x)
        if self.use_logsoftmax:
            x = self.logsoftmax(x.logits)
        else:
            x = x.logits

        return x
