import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEncoder(nn.Module):
    def __init__(self, time_encoder_config):
        super().__init__()
        self.config = time_encoder_config

        '''
        time_encoder_config.size             # int
        time_encoder_config.stride           # int
        time_encoder_config.padding          # int
        time_encoder_config.channels         # list of 4 ints (e.g., [1, 24, 32, 40])
        time_encoder_config.activations      # list of 3 activation function names or callables (one per conv layer)
        time_encoder_config.linear_dims      # list of 3 ints (e.g., [640, 100, 2])
        '''

        # Convolutional layers
        c = time_encoder_config.channels  # e.g., [1, 24, 32, 40]
        k, s, p = time_encoder_config.kernel_size, time_encoder_config.stride, time_encoder_config.padding

        self.conv1 = nn.Conv1d(c[0], c[1], kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv1d(c[1], c[2], kernel_size=k, stride=s, padding=p)
        self.conv3 = nn.Conv1d(c[2], c[3], kernel_size=k, stride=s, padding=p)

        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Activation functions (can be strings or callables)
        self.act1 = self._get_activation(time_encoder_config.activations[0])
        self.act2 = self._get_activation(time_encoder_config.activations[1])
        self.act3 = self._get_activation(time_encoder_config.activations[2])

        self.flatten = nn.Flatten()

        # Fully connected layers
        fc_dims = time_encoder_config.linear_dims  # e.g., [640, 100, 2]
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
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class FreqEncoder(nn.Module):
    def __init__(self, freq_encoder_config):
        super().__init__()
        self.config = freq_encoder_config

        '''
        freq_encoder_config.kernel_size      # int
        freq_encoder_config.stride           # int
        freq_encoder_config.padding          # int
        freq_encoder_config.channels         # list of 4 ints (e.g., [1, 24, 32, 40])
        freq_encoder_config.activations      # list of 3 activation function names or callables
        freq_encoder_config.linear_dims      # list of 3 ints (e.g., [320, 100, 2])
        '''

        # Convolutional layer parameters
        c = freq_encoder_config.channels  # e.g., [1, 24, 32, 40]
        k, s, p = freq_encoder_config.kernel_size, freq_encoder_config.stride, freq_encoder_config.padding

        self.conv1 = nn.Conv1d(c[0], c[1], kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv1d(c[1], c[2], kernel_size=k, stride=s, padding=p)
        self.conv3 = nn.Conv1d(c[2], c[3], kernel_size=k, stride=s, padding=p)

        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Activations
        self.act1 = self._get_activation(freq_encoder_config.activations[0])
        self.act2 = self._get_activation(freq_encoder_config.activations[1])
        self.act3 = self._get_activation(freq_encoder_config.activations[2])

        self.flatten = nn.Flatten()

        # Fully connected layers
        fc_dims = freq_encoder_config.linear_dims  # e.g., [320, 100, 2]
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

def nt_xent_loss(Zt, Zf, temperature=0.5):
    """
    Compute the NT-Xent loss between time and frequency embeddings.
    Zt: (N, d) - time embeddings
    Zf: (N, d) - frequency embeddings
    """
    N = Zt.shape[0]
    z = torch.cat([Zt, Zf], dim=0)  # shape: (2N, d)

    # Normalize for cosine similarity
    z = F.normalize(z, p=2, dim=1)

    # Compute cosine similarity matrix
    sim = torch.matmul(z, z.T)  # shape: (2N, 2N)
    sim = sim / temperature

    # Mask to remove self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pair indices
    positives = torch.cat([
        torch.arange(N, 2*N),
        torch.arange(0, N)
    ]).to(z.device)

    # Compute loss
    loss = F.cross_entropy(sim, positives)
    return loss