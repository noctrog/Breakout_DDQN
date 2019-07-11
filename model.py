import math as m
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class NoisyLinear(nn.Linear):

    """Fully connected layer with gaussian noise. The layer learns the optimal mean and variance of
    the noise. This serves as a substitute to the epsilon-greedy approach to make the model explore
    new outcomes"""

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        """Creates a fully connected layer, with gaussian noise

        :in_features: number of input features
        :out_features: number of output features
        :sigma_init: initial variance value
        :bias: add bias or not

        """
        nn.Linear.__init__(self, in_features, out_features, bias=bias)

        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))

        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        std = m.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, x):
        self.epsilon_weight.normal_()

        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        return F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)


class DQN(nn.Module):

    """Deep Q network used to play breakout"""

    def __init__(self, input_shape, n_actions):
        """Inits the Deep Q network, given input shape and number of actions that the agent can
        perform.

        :input_shape: Shape of 3D matrix (color, x, y)
        :n_actions: number of actions the agent can make in the environment

        """
        nn.Module.__init__(self)

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out_size(input_shape)
        self.classifier = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions)
        )

    def _get_conv_out_size(self, shape):
        """Returns the total number of logits (to connect the convolutional layer with the
        fully-connected layer)

        :x: Input sample
        :returns: No of logits

        """
        out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(out.size()))

    def forward(self, x):
        """Forward pass

        :x: Batch

        """

        x = self.conv(x).view(x.size()[0], -1)
        return self.classifier(x)
