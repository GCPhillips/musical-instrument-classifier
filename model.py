import math
from collections import OrderedDict

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_output=11):
        super().__init__()
        self.conv1 = CNNBlock(1, 32, 3, 2, 2)
        self.conv2 = CNNBlock(32, 64, 3, 2, 2)
        self.conv3 = CNNBlock(64, 128, 3, 2, 2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(12672, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear(x)

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


class InstrumentClassifier(nn.Module):
    def __init__(self, d_input=132299, d_model=4410, d_internal=512, num_classes=11, num_layers=3):
        """
        :param d_input: the dimensions of the classifier input
        :param d_model: the dimensions of the input and output of the transformer
        :param d_internal: the dimensions of the hidden layer in self-attention
        :param num_classes: the number of classes to predict (i.e. the number of musical instruments to classify)
        :param num_layers: the number of transformer layers to use
        """
        super().__init__()
        self.seq_length = d_input

        layers = OrderedDict()
        layers['initial'] = nn.Linear(d_input, d_model)
        for i in range(num_layers):
            layers[f'transformer_layer_{str(i)}'] = TransformerLayer(d_model, d_internal)
        self.layers = nn.Sequential(layers)
        self.prediction = nn.Linear(d_model, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # return self.sigmoid(self.prediction(self.layers(x)))
        return self.prediction(self.layers(x))


class TransformerLayer(nn.Module):
    """
    A single layer of a transformer which includes self attention and a feedforward layer with residual connections.
    """
    def __init__(self, d_model, d_internal):
        """
        :param d_model: the dimensions of the input and output of the transformer layer
        :param d_internal: the hidden layer dimension size
        """
        super().__init__()
        self.attention = SelfAttention(d_model)
        self.ffn = torch.nn.Sequential(
            nn.Linear(d_model, d_internal),
            nn.Tanh(),
            nn.Linear(d_internal, d_model)
        )

    def forward(self, x):
        attention = self.attention(x)
        attention += x
        ffn = self.ffn(attention)
        ffn += attention

        return ffn


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        """
        :param d_model: dimension of the attention layer
        """
        super().__init__()
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.Softmax = nn.Softmax(dim=1)
        self.d_k = math.sqrt(d_model)

    def forward(self, seq):
        q, k, v = self.Q(seq), self.K(seq), self.V(seq)
        num = torch.matmul(q, k.T)
        a = num/self.d_k
        return torch.matmul(self.Softmax(a), v)
