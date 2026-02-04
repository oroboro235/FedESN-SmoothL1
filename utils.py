import numpy as np

from scipy.special import logsumexp as sci_lse


def logsumexp(b, axis=1):
    """
    Computes logsumexp across columns
    """
    # B = np.max(b, axis=1)
    # repmat_B = np.tile(B, (b.shape[1], 1)).T
    # lse = np.log(np.sum(np.exp(b - repmat_B), axis=1)) + B
    return sci_lse(b, axis=axis)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft

# FFT
def get_fft_curves_split_n(x, n=3, max_freq=None, is_plot=False):
    x = x.squeeze()
    fft_result = rfft(x)
    t = np.arange(len(x))
    freqs = rfftfreq(len(x), 1/len(x))

    if max_freq is None or max_freq > np.max(freqs):
        max_freq = np.max(freqs)

    step_size = max_freq / n

    # split the signal into n parts
    results = []
    for i in range(n):
        start_freq = i * step_size
        # end_freq = start_freq + (int)(step_size * 1.5)
        end_freq = start_freq + step_size
        freq_indices = np.where((freqs >= start_freq) & (freqs < end_freq))[0]
        if len(freq_indices) == 0:
            continue
        freq_result = np.zeros_like(fft_result)
        freq_result[freq_indices] = fft_result[freq_indices]
        signal = irfft(freq_result, n=len(t))
        results.append(signal.reshape(-1, 1))

        if is_plot:
            plt.figure(figsize=(12, 8))
            plt.plot(t, signal, label=f'{start_freq:.2f} - {end_freq:.2f} Hz')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.grid(True)
            plt.xlabel('Time')
            plt.title(f'FFT analysis, {n} parts')
            plt.show()

    # plot the difference between reconstructed signal and original signal
    if is_plot:
        plt.figure(figsize=(12, 8))
        plt.plot(t, x, label='Original signal')
        plt.plot(t, np.sum(results, axis=0), label='Reconstructed signal')
        plt.legend()
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.title('FFT analysis')
        plt.show()

    return results







def get_fft_curves_top_n(x, top_n=3):
    x = x.squeeze()
    fft_result = rfft(x)
    t = np.arange(len(x))
    freqs = rfftfreq(len(x), 1/len(x))

    mag = np.abs(fft_result)
    non_zero_indices = freqs > 0
    
    top_indices = np.argsort(mag[non_zero_indices])[-top_n:][::-1]
    top_freqs = freqs[non_zero_indices][top_indices]
    top_mag = mag[non_zero_indices][top_indices]

    print(f"Found main frequencies: {top_freqs} Hz")
    print(f"Corresponding magnitudes: {top_mag}")

    individual_signals = []
    for i, freq_idx in enumerate(top_indices):
        # Create a spectrum with only the given frequency
        single_fft = np.zeros_like(fft_result)
        # Get the actual index (consider non-zero frequencies)
        actual_idx = np.where(freqs == freqs[non_zero_indices][freq_idx])[0][0]
        single_fft[actual_idx] = fft_result[actual_idx]
        # Symmetric negative frequency component (for real signal)

        if actual_idx < len(fft_result) - 1:
            single_fft[actual_idx + 1] = fft_result[actual_idx + 1]
        
        # Time domain signal is obtained by inverse Fourier transform
        single_signal = irfft(single_fft, n=len(t))
        individual_signals.append(single_signal)

    # add up all the signals to get the final signal, compared with original signal
    final_signal = np.sum(individual_signals, axis=0)
    plt.plot(t, x, label='Original signal')
    plt.plot(t, final_signal, label='Final signal')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title('FFT analysis')
    plt.show()

    return individual_signals


def get_diff_sampling_rate_series(n=5, sample_period=[], len_forecast=1, len_window=100, step=1, n_samples=100,):
    if sample_period == []:
        sample_period = [1] * n

    
    len_ts = (len_window + step * n_samples) * max(sample_period)

    from reservoirpy.datasets import mackey_glass

    raw = mackey_glass(len_ts)
    raw = 2 * (raw - raw.min()) / (raw.max() - raw.min()) - 1


    
    diff_sampling_rate_series = []
    for i in range(n):
        diff_sampling_rate_series.append(raw[::sample_period[i]][:(len_window+step*n_samples)])

    diff_sampling_rate_series = np.array(diff_sampling_rate_series)

    return diff_sampling_rate_series, raw

    




def plot_readout(readout):
    import matplotlib.pyplot as plt
    Wout = readout.Wout
    # bias = readout.bias
    # Wout = np.r_[bias[..., np.newaxis], Wout]

    # calculate the sparsity of the readout
    sparsity = (Wout == 0).mean() * 100
    print(f"Sparsity of the readout: {sparsity:.2f}%")

    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(111)
    ax.grid(axis="y")
    ax.set_ylabel("Coefs. of $W_{out}$")
    ax.set_xlabel("reservoir neurons index")
    ax.bar(np.arange(Wout.size), Wout.ravel()[::-1])

    plt.show()
    plt.savefig("readout_coefs.png")