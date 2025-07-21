import torch
import random

class CreateInputs():
    def __init__(self, global_config):
        """
        Initializes the augmentation helper with global configuration.

        Parameters:
        - global_config: Object containing settings like window_size, time_aug_methods, and freq_aug_methods.
        """
        self.window_size = global_config.dataset_config.window_size

        # Time-domain augmentation parameters
        self.jitter_sigma = global_config.augmentation_config.jitter_sigma
        self.scaling_sigma = global_config.augmentation_config.scaling_sigma

        # Frequency-domain augmentation parameters
        self.remove_segment = global_config.augmentation_config.remove_segment
        self.remove_n_signals = global_config.augmentation_config.remove_n_signals
        self.add_segment = global_config.augmentation_config.add_segment
        self.add_n_peaks = global_config.augmentation_config.add_n_peaks
        self.add_mean_peak = global_config.augmentation_config.add_mean_peak
        self.add_rand_error = global_config.augmentation_config.add_rand_error

        self.time_aug_methods = [
            lambda x: self.jitter(x),
            lambda x: self.scaling(x),
            lambda x: self.permutation(x),
            lambda x: self.masking(x)
        ]

        self.freq_aug_methods = [
            lambda x: self.remove_frequency(x),
            lambda x: self.add_frequency(x)
        ]

    def augment(self, x_time_domain):
        """
        Applies either a time or frequency augmentation to the input signal.

        Parameters:
        - x_time_domain (torch.Tensor): 1D tensor of shape [window_size], original signal.

        Returns:
        - x_original_time (torch.Tensor): original input in time domain
        - x_augmented_time (torch.Tensor): augmented version in time domain
        - x_original_freq (torch.Tensor): FFT amplitude of original signal
        - x_augmented_freq (torch.Tensor): FFT amplitude of augmented signal
        """
        if not isinstance(x_time_domain, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if x_time_domain.dim() != 1 or x_time_domain.shape[0] != self.window_size:
            raise ValueError(f"Input must be a 1D tensor of length {self.window_size}.")

        # Save original and subtract mean
        x_mean = x_time_domain.mean()
        x_centered = x_time_domain - x_mean
        x_original_freq = torch.abs(torch.fft.rfft(x_centered))

        # Decide augmentation type
        if random.choice(["time", "freq"]) == "time":
            aug_method = random.choice(self.time_aug_methods)
            x_aug_time = aug_method(x_time_domain.clone())  # may include its own mean adjustment
            x_aug_freq = torch.abs(torch.fft.rfft(x_aug_time - x_aug_time.mean()))

            return x_time_domain, x_aug_time, x_original_freq, x_aug_freq

        else:
            # Frequency-domain augmentation
            x_freq_aug_func = random.choice(self.freq_aug_methods)
            x_aug_freq = x_freq_aug_func(x_original_freq.clone())

            # Reconstruct time-domain signal
            x_aug_time = torch.fft.irfft(x_aug_freq, n=self.window_size) + x_mean

            return x_time_domain, x_aug_time, x_original_freq, x_aug_freq

    ########################################
    ########################################
    # Data Augmentation - Time series
    #########################################
    #########################################


    def jitter(x):

        sigma=0.6
        """
        Applies jitter augmentation to the input data.
        
        Parameters:
        - x: Input data.
        - sigma: Standard deviation of the jitter noise (default is 0.6).
        
        Returns:
        - Jittered data.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch tensor, but got {type(x)}.")
        return x + torch.randn_like(x) * sigma

    def scaling(x):
        sigma=1.1
        """
        Applies scaling augmentation to the input data. The scaling factor is drawn from a normal distribution
        and multiplies each value in the time series.

        Parameters:
        - x: Input data (torch tensor) with shape (num_sensors, num_timesteps).
        - sigma: Standard deviation of the scaling factor (default is 1.1).

        Returns:
        - Scaled data (torch tensor) with the same shape as the input.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch tensor, but got {type(x)}.")
        
        scaling_factor = torch.normal(mean=2.0, std=sigma, size=(1, x.shape[1]), device=x.device)
        return x * scaling_factor

    def permutation(x):
        seg_mode="random"
        max_segments=5
        """
        Applies permutation augmentation to the input data. The time series is split into segments and the segments are randomly permuted.
        
        Parameters:
        - x: Input data tensor (time series).
        - max_segments: Maximum number of segments to split the time series into (default is 5).
        - seg_mode: Mode of segmentation. Can be "random" for random splits or "equal" for equal splits (default is "random").
        
        Returns:
        - permuted_data: Permuted time series data.
        - affected_indices: Indices of the original time series that were permuted.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch tensor, but got {type(x)}.")
        
        orig_steps = torch.arange(x.size(0))
        num_segs = min(max_segments, x.size(0))

        if seg_mode == "random":
            split_points = torch.randint(1, x.size(0), (num_segs - 1,))
            split_points, _ = torch.sort(split_points)
        else:
            split_points = torch.arange(1, x.size(0), x.size(0) // num_segs)[:num_segs - 1]

        segments = torch.split(orig_steps, torch.diff(torch.cat((torch.tensor([0]), split_points, torch.tensor([x.size(0)])))).tolist())
        permuted_segments = [segments[i] for i in torch.randperm(len(segments))]
        permuted_indices = torch.cat(permuted_segments)

        return x[permuted_indices]

    def masking(x):
        mask='binomial'
        keepratio=0.9
        """
        Applies masking to the input data. The selected values are set to zero.
        
        Parameters:
        - x: Input data array (row vector).
        - keepratio: The ratio of values to keep (default is 0.9).
        - mask: The type of mask to use (default is 'binomial').
        
        Returns:
        - Masked data array and affected indices.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch tensor, but got {type(x)}.")
        x2 = x.clone()
        nan_mask = ~torch.isnan(x)
        x2[~nan_mask] = 0
        
        if mask == 'binomial':
            mask_id = torch.bernoulli(torch.full_like(x, keepratio)).bool()
        else:
            raise ValueError(f"Mask type '{mask}' not supported.")
        
        x2[~mask_id] = 0

        return x2

        
    ########################################
    ########################################
    # Data Augmentation - Frequency domain
    #########################################
    #########################################

    def remove_frequency(amplitude_spectrum):
        remove_ratio=0.0
        """
        Removes a percentage of amplitude components from the input amplitude spectrum.

        Parameters:
        - amplitude_spectrum: Real-valued amplitude spectrum (half spectrum).
        - remove_ratio: Ratio of components to remove.

        Returns:
        - Modified amplitude spectrum.
        """
        if not isinstance(amplitude_spectrum, torch.Tensor):
            raise TypeError(f"Input must be a torch tensor, but got {type(amplitude_spectrum)}.")
        if torch.is_complex(amplitude_spectrum):
            raise ValueError(f"Input amplitude spectrum must be real-valued, but got complex values.")
        mask = torch.rand_like(amplitude_spectrum) > remove_ratio
        return amplitude_spectrum * mask

    def add_frequency(amplitude_spectrum):
        const=0.2
        add_ratio=0.0
        """
        Adds random frequency components to the input signal. If the random number is greater than (1 - perturb_ratio), we get a list of
        the frequency components. From this list, we get the maximum amplitude and use it to define a uniform
        distribution. The scaling factor alpha is sampled from this distribution and multiplied by the mask. These results in a list
        of 0 and scaled values. We add this list to the input signal.
        
        Larger perturb_ratio values will result in more frequency components being added.
        
        Parameters:
        - x: Input signal (frequency domain).
        - perturb_ratio: Ratio of frequency components to add. 1.0 means we perturb all frequency components, 0.0 means we perturb nothing.
        
        Returns:
        - Signal with added frequency components.
        """
        if not isinstance(amplitude_spectrum, torch.Tensor):
            raise TypeError(f"Input must be a torch tensor, but got {type(amplitude_spectrum)}.")
        if amplitude_spectrum.is_complex():
            raise ValueError(f"Input amplitude spectrum must be real-valued, but got complex values.")
        
        mask = torch.rand_like(amplitude_spectrum) > add_ratio
        max_amplitude = torch.max(amplitude_spectrum)
        alpha = torch.rand_like(amplitude_spectrum) * max_amplitude * const
        return amplitude_spectrum + mask * alpha