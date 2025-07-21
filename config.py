class AugmentationConfig:
    """
    Configuration for data augmentation parameters.

    Parameters:
    - jitter_sigma (float): Std. deviation for time-domain jitter augmentation. Default None.
    - scaling_sigma (float): Std. deviation for time-domain scaling augmentation. Default None.
    - remove_segment (int): Segment index for frequency removal augmentation. Default None.
    - remove_n_signals (int): Number of frequency signals to remove. Default None.
    - add_segment (int): Segment index for frequency addition augmentation. Default None.
    - add_n_peaks (int): Number of frequency peaks to add. Default None.
    - add_mean_peak (float): Mean amplitude of added peaks. Default None.
    - add_rand_error (float): Random variation range around mean_peak. Default None.
    """
    def __init__(self,
                 jitter_sigma=3,
                 scaling_sigma=3,
                 remove_segment=1,
                 remove_n_signals=3,
                 add_segment=1,
                 add_n_peaks=3,
                 add_mean_peak=80,
                 add_rand_error=10):
        
        self.jitter_sigma = jitter_sigma
        self.scaling_sigma = scaling_sigma
        self.remove_segment = remove_segment
        self.remove_n_signals = remove_n_signals
        self.add_segment = add_segment
        self.add_n_peaks = add_n_peaks
        self.add_mean_peak = add_mean_peak
        self.add_rand_error = add_rand_error


class TimeEncoderConfig:
    """
    Configuration for the time-domain encoder.

    Parameters:
    - kernel_size (list of int): Input size of the kernels.
    - stride (list of int): Stride for convolutional layers.
    - channels (list of int): Channel sizes for conv layers, length 4 (input + 3 layers).
    - activations (list of str): Activation function names for each conv layer.
    - linear_dims (list of int): Dimensions for the fully connected layers, length 3.
    """
    def __init__(self,
                 kernel_size=[8, 8, 8, 4],
                 stride=[4, 4, 4, 2],
                 channels=[1, 2, 3, 4, 5],
                 activations=["relu", "relu", "relu", "relu"],
                 linear_dims=[40, 10, 2]):
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.activations = activations
        self.linear_dims = linear_dims


class FreqEncoderConfig:
    """
    Configuration for the frequency-domain encoder.

    Parameters:
    - kernel_size (int): Kernel size for convolutional layers.
    - stride (int): Stride for convolutional layers.
    - padding (int): Padding for convolutional layers.
    - channels (list of int): Channel sizes for conv layers, length 4 (input + 3 layers).
    - activations (list of str): Activation function names for each conv layer.
    - linear_dims (list of int): Dimensions for the fully connected layers, length 3.
    """
    def __init__(self,
                 kernel_size=[8, 4, 4],
                 stride=[4, 2, 2],
                 channels=[1, 2, 3, 4],
                 activations=["relu", "relu", "relu"],
                 linear_dims=[32, 8, 2]):
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.activations = activations
        self.linear_dims = linear_dims

class DatasetConfig:
    def __init__(self, 
                 input_df,
                 sensor_column='UA_Z_AL',
                 window_size=128, 
                 step_size=32, 
                 max_time_gap=2):

        self.input_df = input_df
        self.window_size = window_size
        self.step_size = step_size
        self.max_time_gap = max_time_gap
        self.sensor_column = sensor_column

class GlobalConfig:
    def __init__(self, augmentation_config=None, time_encoder_config=None, freq_encoder_config=None, dataset_config=None):
        self.augmentation_config = augmentation_config if augmentation_config is not None else AugmentationConfig()
        self.time_encoder_config = time_encoder_config if time_encoder_config is not None else TimeEncoderConfig()
        self.freq_encoder_config = freq_encoder_config if freq_encoder_config is not None else FreqEncoderConfig()
        if dataset_config is None:
            raise ValueError("Input df not defined.")
        self.dataset_config = dataset_config
