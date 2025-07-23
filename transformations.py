import torch
import random
import numpy as np

class CreateInputs:
    def __init__(self, global_config):
        """
        Initializes the augmentation helper with global configuration.

        Parameters:
        - global_config: Object containing settings like window_size, time_aug_methods, and freq_aug_methods.
        """
        self.config = global_config
        self.window_size = global_config.dataset_config.window_size

        # Time-domain augmentation parameters
        self.jitter_sigma = self.config.augmentation_config.jitter_sigma
        self.scaling_sigma = self.config.augmentation_config.scaling_sigma

        # Frequency-domain augmentation parameters
        self.remove_segment = self.config.augmentation_config.remove_segment
        self.remove_n_signals = self.config.augmentation_config.remove_n_signals
        self.add_segment = self.config.augmentation_config.add_segment
        self.add_n_peaks = self.config.augmentation_config.add_n_peaks
        self.add_mean_peak = self.config.augmentation_config.add_mean_peak
        self.add_rand_error = self.config.augmentation_config.add_rand_error

    def __call__(self, x_time_domain_batch, augment=True):
        return self.augment_batch(x_time_domain_batch, augment=augment)

    def augment_batch(self, x_time_domain_batch, augment=True):
        """
        Applies augmentations to a batch of signals efficiently.

        Parameters:
        - x_time_domain_batch (torch.Tensor): Tensor of shape [batch_size, window_size]
        - augment (bool): Whether to apply augmentations or not
        If augment is False, returns original signals without modifications.
        Returns:
        - x_original_time (torch.Tensor): original inputs in time domain [batch_size, window_size]
        - x_augmented_time (torch.Tensor): augmented versions in time domain [batch_size, window_size]
        - x_original_freq (torch.Tensor): FFT amplitude of original signals [batch_size, freq_bins]
        - x_augmented_freq (torch.Tensor): FFT amplitude of augmented signals [batch_size, freq_bins]
        """
        if not isinstance(x_time_domain_batch, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        
        if x_time_domain_batch.dim() != 2:
            raise ValueError("Input must be a 2D tensor of shape [batch_size, window_size].")
        
        if x_time_domain_batch.shape[1] != self.window_size:
            raise ValueError(f"Window size must be {self.window_size}, got {x_time_domain_batch.shape[1]}.")

        batch_size = x_time_domain_batch.shape[0]
        device = x_time_domain_batch.device
        
        # Center the signals (subtract mean along window dimension)
        x_means = x_time_domain_batch.mean(dim=1, keepdim=True)
        x_centered = x_time_domain_batch - x_means
        
        # Compute original frequency domain representations
        x_original_freq = torch.abs(torch.fft.rfft(x_centered, dim=1))

        if not augment:
            # If augmentation is disabled, return original signals and their frequency representations
            return x_time_domain_batch, x_original_freq.clone()
        
        # Randomly decide augmentation type for each sample in batch
        aug_types = torch.randint(0, 2, (batch_size,), device=device)  # 0=time, 1=freq
        time_mask = aug_types == 0
        freq_mask = aug_types == 1
        
        # Initialize augmented outputs
        x_aug_time = x_time_domain_batch.clone()
        x_aug_freq = x_original_freq.clone()
        
        # Apply time-domain augmentations to selected samples
        if time_mask.any():
            time_indices = torch.where(time_mask)[0]
            x_time_subset = x_time_domain_batch[time_indices]
            
            # Randomly choose time augmentation method for each sample
            time_methods = torch.randint(0, 2, (len(time_indices),), device=device)  # 0=jitter, 1=scaling
            
            # Apply jitter to selected samples
            jitter_mask = time_methods == 0
            if jitter_mask.any():
                jitter_indices = time_indices[jitter_mask]
                x_aug_time[jitter_indices] = self.jitter_batch(x_time_subset[jitter_mask])
            
            # Apply scaling to selected samples
            scaling_mask = time_methods == 1
            if scaling_mask.any():
                scaling_indices = time_indices[scaling_mask]
                x_aug_time[scaling_indices] = self.scaling_batch(x_time_subset[scaling_mask])
            
            # Recompute frequency domain for time-augmented samples
            x_aug_centered = x_aug_time[time_indices] - x_aug_time[time_indices].mean(dim=1, keepdim=True)
            x_aug_freq[time_indices] = torch.abs(torch.fft.rfft(x_aug_centered, dim=1))
        
        # Apply frequency-domain augmentations to selected samples
        if freq_mask.any():
            freq_indices = torch.where(freq_mask)[0]
            x_freq_subset = x_original_freq[freq_indices]
            
            # Randomly choose frequency augmentation method for each sample
            freq_methods = torch.randint(0, 2, (len(freq_indices),), device=device)  # 0=remove, 1=add
            
            # Apply frequency removal to selected samples
            remove_mask = freq_methods == 0
            if remove_mask.any():
                remove_indices = freq_indices[remove_mask]
                x_aug_freq[remove_indices] = self.remove_frequency_batch(x_freq_subset[remove_mask])
            
            # Apply frequency addition to selected samples
            add_mask = freq_methods == 1
            if add_mask.any():
                add_indices = freq_indices[add_mask]
                x_aug_freq[add_indices] = self.add_frequency_batch(x_freq_subset[add_mask])
            
            # Reconstruct time-domain signals for frequency-augmented samples
            x_aug_time_centered = torch.fft.irfft(x_aug_freq[freq_indices], n=self.window_size, dim=1)
            x_aug_time[freq_indices] = x_aug_time_centered + x_means[freq_indices]
        
        return x_time_domain_batch, x_aug_time, x_original_freq, x_aug_freq

    ########################################
    ########################################
    # Data Augmentation - Time series
    #########################################
    #########################################
    def jitter_batch(self, x_batch, sigma=None):
        """Apply jitter augmentation to a batch of signals."""
        if sigma is None:
            sigma = self.jitter_sigma
        
        noise = torch.randn_like(x_batch) * sigma
        return x_batch + noise
    
    def scaling_batch(self, x_batch, sigma=None):
        """Apply scaling augmentation to a batch of signals."""
        if sigma is None:
            sigma = self.scaling_sigma
        
        batch_size = x_batch.shape[0]
        device = x_batch.device
        
        # Generate random scaling factors for each sample
        scaling_factors = torch.normal(mean=1.0, std=sigma, size=(batch_size, 1), device=device)
        return x_batch * scaling_factors
    
    def remove_frequency_batch(self, x_freq_batch, segment=None, n_signals=None):
        """Remove frequency components from a batch of frequency domain signals."""
        if segment is None:
            segment = self.remove_segment
        if n_signals is None:
            n_signals = self.remove_n_signals
        
        batch_size, freq_bins = x_freq_batch.shape
        device = x_freq_batch.device
        
        x_aug = x_freq_batch.clone()
        
        # For each sample in batch, randomly remove frequency segments
        for i in range(batch_size):
            for _ in range(n_signals):
                start_idx = torch.randint(0, max(1, freq_bins - segment), (1,), device=device).item()
                end_idx = min(start_idx + segment, freq_bins)
                x_aug[i, start_idx:end_idx] = 0
        
        return x_aug
    
    def add_frequency_batch(self, x_freq_batch, segment=None, n_peaks=None, mean_peak=None, rand_error=None):
        """Add frequency components to a batch of frequency domain signals."""
        if segment is None:
            segment = self.add_segment
        if n_peaks is None:
            n_peaks = self.add_n_peaks
        if mean_peak is None:
            mean_peak = self.add_mean_peak
        if rand_error is None:
            rand_error = self.add_rand_error
        
        batch_size, freq_bins = x_freq_batch.shape
        device = x_freq_batch.device
        
        x_aug = x_freq_batch.clone()
        
        # For each sample in batch, randomly add frequency peaks
        for i in range(batch_size):
            for _ in range(n_peaks):
                start_idx = torch.randint(0, max(1, freq_bins - segment), (1,), device=device).item()
                end_idx = min(start_idx + segment, freq_bins)
                
                # Generate peak with some randomness
                peak_amplitude = mean_peak + torch.randn(1, device=device).item() * rand_error
                peak_amplitude = max(0, peak_amplitude)  # Ensure non-negative
                
                # Add triangular or gaussian-like peak
                peak_length = end_idx - start_idx
                if peak_length > 0:
                    # Create a simple triangular peak
                    peak_indices = torch.arange(peak_length, device=device, dtype=torch.float32)
                    peak_center = peak_length / 2
                    peak_values = peak_amplitude * (1 - torch.abs(peak_indices - peak_center) / peak_center)
                    peak_values = torch.clamp(peak_values, min=0)
                    
                    x_aug[i, start_idx:end_idx] += peak_values
        
        return x_aug
    
    def augment(self, x_time_domain):
        """
        Legacy method for single sample processing (for backward compatibility).
        Uses the batch method internally for consistency.
        """
        if x_time_domain.dim() == 1:
            x_time_domain = x_time_domain.unsqueeze(0)  # Add batch dimension
        
        x_orig, x_aug, x_orig_freq, x_aug_freq = self.augment_batch(x_time_domain)
        
        # Remove batch dimension for single sample
        return x_orig.squeeze(0), x_aug.squeeze(0), x_orig_freq.squeeze(0), x_aug_freq.squeeze(0)


    # def jitter(self, x_time, sigma=0.6):
    #     """
    #     Applies jitter augmentation to the input data.
        
    #     Parameters:
    #     - x_time: Input data.
    #     - sigma: Standard deviation of the jitter noise (default is 0.6).
        
    #     Returns:
    #     - Jittered data.
    #     """
    #     if not isinstance(x_time, torch.Tensor):
    #         raise TypeError(f"Input must be a torch tensor, but got {type(x_time)}.")
    #     return x_time + torch.randn_like(x_time) * sigma

    # def scaling(self, x_time, sigma=1.1):
    #     """
    #     Applies scaling augmentation to the input data. The scaling factor is drawn from a normal distribution
    #     and multiplies each value in the time series.

    #     Parameters:
    #     - x_time: Input data (torch tensor) with shape (num_sensors, num_timesteps).
    #     - sigma: Standard deviation of the scaling factor (default is 1.1).

    #     Returns:
    #     - Scaled data (torch tensor) with the same shape as the input.
    #     """
    #     if not isinstance(x_time, torch.Tensor):
    #         raise TypeError(f"Input must be a torch tensor, but got {type(x_time)}.")
        
    #     scaling_factor = torch.normal(mean=2.0, std=sigma, size=(1, x_time.shape[1]), device=x_time.device)
    #     return x_time * scaling_factor
        
    # ########################################
    # ########################################
    # # Data Augmentation - Frequency domain #
    # ########################################
    # ########################################

    # def remove_frequency(self, x_freq, segment=1, n_signals=1):
    #     """
    #     Suppresses selected frequency components from a specified segment of the FFT amplitude spectrum
    #     by setting their amplitudes to zero.

    #     Parameters:
    #     - x_freq (torch.Tensor): 1D real-valued tensor representing the amplitude spectrum (e.g., from rFFT).
    #     - segment (int): Segment of the spectrum where frequencies will be removed.
    #                     Values: 1=0-25%, 2=25-50%, 3=50-75%, 4=75-100%.
    #     - n_signals (int): Number of frequency bins to zero-out in the selected segment.

    #     Returns:
    #     - torch.Tensor: Modified amplitude spectrum with selected frequencies removed.
    #     """
    #     if not isinstance(x_freq, torch.Tensor):
    #         raise TypeError(f"Input must be a torch tensor, but got {type(x_freq)}.")
    #     if x_freq.is_complex():
    #         raise ValueError("Input amplitude spectrum must be real-valued, not complex.")
    #     if segment not in [1, 2, 3, 4]:
    #         raise ValueError("Segment must be an integer between 1 and 4.")

    #     spectrum = x_freq.clone()
    #     n = spectrum.shape[0]

    #     # Define segment bounds
    #     start = (segment - 1) * (n // 4)
    #     end = segment * (n // 4)

    #     # Ensure number of removed signals doesn't exceed available positions
    #     available_indices = torch.arange(start, end)
    #     n_signals = min(n_signals, len(available_indices))

    #     # Randomly select which frequencies to zero out
    #     remove_positions = available_indices[torch.randperm(len(available_indices))[:n_signals]]

    #     # Zero out selected frequencies
    #     spectrum[remove_positions] = 0.0

    #     return spectrum

    # def add_frequency(self, x_freq, segment=1, n_peaks=1, mean_peak=80, rand_error=10):
    #     """
    #     Injects artificial peaks into the FFT amplitude spectrum of a signal to simulate frequency-domain anomalies.
        
    #     A specific quarter segment of the spectrum is selected (low, mid-low, mid-high, or high frequency). 
    #     Within this segment, `n_peaks` frequencies are randomly chosen and augmented with peaks whose amplitudes are 
    #     sampled from a uniform distribution centered at `mean_peak` with a spread of ±`rand_error`.

    #     Parameters:
    #     - x_freq (torch.Tensor): 1D real-valued tensor representing the amplitude spectrum (e.g., from rFFT).
    #     - segment (int): Segment of the spectrum where peaks will be injected.
    #                     Values: 1=0-25%, 2=25-50%, 3=50-75%, 4=75-100%.
    #     - n_peaks (int): Number of peaks to inject into the spectrum.
    #     - mean_peak (float): Mean amplitude value of the injected peaks.
    #     - rand_error (float): Range for random variation around `mean_peak`. Injected amplitude ∈ [mean_peak - rand_error, mean_peak + rand_error].

    #     Returns:
    #     - torch.Tensor: Augmented amplitude spectrum (same shape as input).
    #     """
    #     if not isinstance(x_freq, torch.Tensor):
    #         raise TypeError(f"Input must be a torch tensor, but got {type(x_freq)}.")
    #     if x_freq.is_complex():
    #         raise ValueError("Input amplitude spectrum must be real-valued, not complex.")
    #     if segment not in [1, 2, 3, 4]:
    #         raise ValueError("Segment must be an integer between 1 and 4.")
        
    #     spectrum = x_freq.clone()
    #     n = spectrum.shape[0]

    #     # Define segment bounds
    #     start = (segment - 1) * (n // 4)
    #     end = segment * (n // 4)
        
    #     # Ensure number of peaks doesn't exceed available positions
    #     available_indices = torch.arange(start, end)
    #     n_peaks = min(n_peaks, len(available_indices))
        
    #     # Randomly select peak positions
    #     peak_positions = available_indices[torch.randperm(len(available_indices))[:n_peaks]]

    #     # Generate random amplitudes
    #     noise_amplitudes = torch.empty(n_peaks).uniform_(mean_peak - rand_error, mean_peak + rand_error)

    #     # Inject peaks
    #     spectrum[peak_positions] += noise_amplitudes

    #     return spectrum