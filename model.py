import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEncoder(nn.Module):
    def __init__(self, time_encoder_config):
        super().__init__()

        self.config = time_encoder_config

        '''
        time_encoder_config.kernel_size      # list of 4 ints (e.g., [8, 8, 8, 4])
        time_encoder_config.stride           # list of 4 ints (e.g., [4, 4, 4, 2])
        time_encoder_config.channels         # list of 5 ints (e.g., [1, 2, 3, 4, 5])
        time_encoder_config.activations      # list of 4 activation function names (e.g., ["relu", "relu", "relu", "relu"])
        time_encoder_config.linear_dims      # list of 3 ints (e.g., [40, 10, 2])
        '''

        # Convolutional layers
        c = self.config.channels  # e.g., [1, 24, 32, 40]
        k, s = self.config.kernel_size, self.config.stride

        self.conv1 = nn.Conv1d(c[0], c[1], kernel_size=k, stride=s, padding=s)
        self.conv2 = nn.Conv1d(c[1], c[2], kernel_size=k, stride=s, padding=s)
        self.conv3 = nn.Conv1d(c[2], c[3], kernel_size=k, stride=s, padding=s)
        self.conv4 = nn.Conv1d(c[3], c[4], kernel_size=k, stride=s, padding=s)

        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        # Activations
        self.act1 = self._get_activation(self.config.activations[0])
        self.act2 = self._get_activation(self.config.activations[1])
        self.act3 = self._get_activation(self.config.activations[2])
        self.act4 = self._get_activation(self.config.activations[3])

        self.flatten = nn.Flatten()

        # Fully connected layers
        fc_dims = self.config.linear_dims  # e.g., [40, 10, 2]
        self.fc1 = nn.Linear(fc_dims[0], fc_dims[1])
        self.fc2 = nn.Linear(fc_dims[1], fc_dims[2])

    def _get_activation(self, act):
        if isinstance(act, str):
            act = act.lower()
            return {
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "gelu": nn.GELU(),
                "leaky_relu": nn.LeakyReLU()
            }.get(act, nn.ReLU())
        return act  # Already a callable/module

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.pool4(self.act4(self.conv4(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class FreqEncoder(nn.Module):
    def __init__(self, freq_encoder_config):
        super().__init__()

        self.config = freq_encoder_config

        '''
        self.config.kernel_size      # list of 3 ints (e.g., [8, 4, 4])
        self.config.stride           # list of 3 ints (e.g., [4, 2, 2])
        self.config.channels         # list of 4 ints (e.g., [1, 2, 3, 4])
        self.config.activations      # list of 3 activation function names (e.g., ["relu", "relu", "relu"])
        self.config.linear_dims      # list of 3 ints (e.g., [32, 8, 2])
        '''

        # Convolutional layer parameters
        c = self.config.channels
        k, s = self.config.kernel_size, self.config.stride

        self.conv1 = nn.Conv1d(c[0], c[1], kernel_size=k, stride=s, padding=s)
        self.conv2 = nn.Conv1d(c[1], c[2], kernel_size=k, stride=s, padding=s)
        self.conv3 = nn.Conv1d(c[2], c[3], kernel_size=k, stride=s, padding=s)

        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.act1 = self._get_activation(self.config.activations[0])
        self.act2 = self._get_activation(self.config.activations[1])
        self.act3 = self._get_activation(self.config.activations[2])

        self.flatten = nn.Flatten()

        # Fully connected layers
        fc_dims = self.config.linear_dims  # e.g., [32, 8, 2]
        self.fc1 = nn.Linear(fc_dims[0], fc_dims[1])
        self.fc2 = nn.Linear(fc_dims[1], fc_dims[2])

    def _get_activation(self, act):
        if isinstance(act, str):
            act = act.lower()
            return {
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "gelu": nn.GELU(),
                "leaky_relu": nn.LeakyReLU()
            }.get(act, nn.ReLU())
        return act

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DualTimeSeriesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = TimeEncoder()
        self.encoder2 = FreqEncoder()

    def forward(self, x1, x2):
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        return z1, z2


def loss(Zt, Zf, Zt_aug, Zf_aug, tau=2, delta=1):

    normal_dist = (torch.norm(Zt-Zf)^2)/tau
    
    augmented_dist = (torch.norm((Zt_aug-Zf_aug))^2)/tau

    pair_dist =  torch.norm(((Zt + Zf)/2) - ((Zt_aug + Zf_aug)/2))
    delta_dist = (abs(pair_dist - delta)^2)/tau

    return normal_dist + augmented_dist + delta_dist