import torch
import random

class create_inputs():
    def __init__(self, global_config):
        """
        Initializes the augmentation helper with global configuration.

        Parameters:
        - global_config: Object containing settings like window_size, time_aug_methods, and freq_aug_methods.
        """
        self.config = global_config
        self.window_size = global_config.window_size

        # Time-domain augmentation parameters
        self.jitter_sigma = self.config.jitter_sigma
        self.scaling_sigma = self.config.scaling_sigma

        # Frequency-domain augmentation parameters
        self.remove_segment = self.config.remove_segment
        self.remove_n_signals = self.config.remove_n_signals
        self.add_segment = self.config.add_segment
        self.add_n_peaks = self.config.add_n_peaks
        self.add_mean_peak = self.config.add_mean_peak
        self.add_rand_error = self.config.add_rand_error

        self.time_aug_methods = [
            lambda x: self.jitter(x, sigma=self.jitter_sigma),
            lambda x: self.scaling(x, sigma=self.scaling_sigma)
        ]

        self.freq_aug_methods = [
            lambda x: self.remove_frequency(x, segment=self.remove_segment, n_signals=self.remove_n_signals),
            lambda x: self.add_frequency(x, segment=self.add_segment, n_peaks=self.add_n_peaks, 
                                    mean_peak=self.add_mean_peak, rand_error=self.add_rand_error)
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


    def jitter(self, x_time, sigma=0.6):
        """
        Applies jitter augmentation to the input data.
        
        Parameters:
        - x_time: Input data.
        - sigma: Standard deviation of the jitter noise (default is 0.6).
        
        Returns:
        - Jittered data.
        """
        if not isinstance(x_time, torch.Tensor):
            raise TypeError(f"Input must be a torch tensor, but got {type(x_time)}.")
        return x_time + torch.randn_like(x_time) * sigma

    def scaling(self, x_time, sigma=1.1):
        """
        Applies scaling augmentation to the input data. The scaling factor is drawn from a normal distribution
        and multiplies each value in the time series.

        Parameters:
        - x_time: Input data (torch tensor) with shape (num_sensors, num_timesteps).
        - sigma: Standard deviation of the scaling factor (default is 1.1).

        Returns:
        - Scaled data (torch tensor) with the same shape as the input.
        """
        if not isinstance(x_time, torch.Tensor):
            raise TypeError(f"Input must be a torch tensor, but got {type(x_time)}.")
        
        scaling_factor = torch.normal(mean=2.0, std=sigma, size=(1, x_time.shape[1]), device=x_time.device)
        return x_time * scaling_factor
        
    ########################################
    ########################################
    # Data Augmentation - Frequency domain #
    ########################################
    ########################################

    def remove_frequency(self, x_freq, segment=1, n_signals=1):
        """
        Suppresses selected frequency components from a specified segment of the FFT amplitude spectrum
        by setting their amplitudes to zero.

        Parameters:
        - x_freq (torch.Tensor): 1D real-valued tensor representing the amplitude spectrum (e.g., from rFFT).
        - segment (int): Segment of the spectrum where frequencies will be removed.
                        Values: 1=0-25%, 2=25-50%, 3=50-75%, 4=75-100%.
        - n_signals (int): Number of frequency bins to zero-out in the selected segment.

        Returns:
        - torch.Tensor: Modified amplitude spectrum with selected frequencies removed.
        """
        if not isinstance(x_freq, torch.Tensor):
            raise TypeError(f"Input must be a torch tensor, but got {type(x_freq)}.")
        if x_freq.is_complex():
            raise ValueError("Input amplitude spectrum must be real-valued, not complex.")
        if segment not in [1, 2, 3, 4]:
            raise ValueError("Segment must be an integer between 1 and 4.")

        spectrum = x_freq.clone()
        n = spectrum.shape[0]

        # Define segment bounds
        start = (segment - 1) * (n // 4)
        end = segment * (n // 4)

        # Ensure number of removed signals doesn't exceed available positions
        available_indices = torch.arange(start, end)
        n_signals = min(n_signals, len(available_indices))

        # Randomly select which frequencies to zero out
        remove_positions = available_indices[torch.randperm(len(available_indices))[:n_signals]]

        # Zero out selected frequencies
        spectrum[remove_positions] = 0.0

        return spectrum

    def add_frequency(self, x_freq, segment=1, n_peaks=1, mean_peak=80, rand_error=10):
        """
        Injects artificial peaks into the FFT amplitude spectrum of a signal to simulate frequency-domain anomalies.
        
        A specific quarter segment of the spectrum is selected (low, mid-low, mid-high, or high frequency). 
        Within this segment, `n_peaks` frequencies are randomly chosen and augmented with peaks whose amplitudes are 
        sampled from a uniform distribution centered at `mean_peak` with a spread of ±`rand_error`.

        Parameters:
        - x_freq (torch.Tensor): 1D real-valued tensor representing the amplitude spectrum (e.g., from rFFT).
        - segment (int): Segment of the spectrum where peaks will be injected.
                        Values: 1=0-25%, 2=25-50%, 3=50-75%, 4=75-100%.
        - n_peaks (int): Number of peaks to inject into the spectrum.
        - mean_peak (float): Mean amplitude value of the injected peaks.
        - rand_error (float): Range for random variation around `mean_peak`. Injected amplitude ∈ [mean_peak - rand_error, mean_peak + rand_error].

        Returns:
        - torch.Tensor: Augmented amplitude spectrum (same shape as input).
        """
        if not isinstance(x_freq, torch.Tensor):
            raise TypeError(f"Input must be a torch tensor, but got {type(x_freq)}.")
        if x_freq.is_complex():
            raise ValueError("Input amplitude spectrum must be real-valued, not complex.")
        if segment not in [1, 2, 3, 4]:
            raise ValueError("Segment must be an integer between 1 and 4.")
        
        spectrum = x_freq.clone()
        n = spectrum.shape[0]

        # Define segment bounds
        start = (segment - 1) * (n // 4)
        end = segment * (n // 4)
        
        # Ensure number of peaks doesn't exceed available positions
        available_indices = torch.arange(start, end)
        n_peaks = min(n_peaks, len(available_indices))
        
        # Randomly select peak positions
        peak_positions = available_indices[torch.randperm(len(available_indices))[:n_peaks]]

        # Generate random amplitudes
        noise_amplitudes = torch.empty(n_peaks).uniform_(mean_peak - rand_error, mean_peak + rand_error)

        # Inject peaks
        spectrum[peak_positions] += noise_amplitudes

        return spectrum