from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

# COMP3710 Demo 2 - Part 1: Discrete Fourier Transform
# Q1-Q8 are our labels mapped to the Part 1 requirements.
# Tags: [Provided] from starter code | [Modified] adapted starter code | [Added] not provided.

# [Provided] Main signal settings from the lab sheet.
N = 2048  # Number of sample points.
T = 1.0  # Signal duration in seconds.
f0 = 1  # Fundamental frequency in Hz.

# [Added] Select CUDA when available for the PyTorch/GPU requirements.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [Added] Save generated figures beside this script.
OUTPUT_DIR = Path(__file__).resolve().parent

# [Added] Print only the hardware information needed to verify GPU availability.
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def save_figure(fig, path):
    """[Added] Save a Matplotlib figure and close it.

    Use: save_figure(fig, path)
    fig: Matplotlib Figure object to save.
    path: output file path.
    """
    fig.savefig(path, dpi=200, bbox_inches="tight")  # Save a high-resolution PNG.
    plt.close(fig)  # Close the figure so WSL does not leave a GUI process open.


# Q1 | Official: run the provided code and plot the square wave and provided harmonics.

def square_wave(t):
    """[Provided] Return an ideal square wave sampled at times t.

    Use: square_wave(t)
    t: NumPy array of time samples in seconds.
    Returns: NumPy array with values -1, 0, or +1.
    """
    # ★ CORE: sign(sin(...)) converts the sine wave into the target square wave.
    return np.sign(np.sin(2.0 * np.pi * f0 * t))


def square_wave_fourier(t, f0, num_harmonics):
    """[Provided] Reconstruct a square wave from odd Fourier harmonics.

    Use: square_wave_fourier(t, f0, num_harmonics)
    t: NumPy array of time samples.
    f0: fundamental frequency in Hz.
    num_harmonics: number of odd harmonics to add.
    Returns: NumPy array containing the reconstructed signal.
    """
    result = np.zeros_like(t)  # Store the accumulated Fourier approximation.
    for k in range(num_harmonics):  # k selects each odd harmonic term.
        n = 2 * k + 1  # ★ CORE: generate harmonic numbers 1, 3, 5, 7, ...
        result += np.sin(2.0 * np.pi * n * f0 * t) / n  # ★ CORE: add harmonic n with amplitude 1/n.
    return (4.0 / np.pi) * result  # Apply the square-wave Fourier-series scale factor.


t = np.linspace(0.0, T, N, endpoint=False)  # [Provided] Create N periodic time samples.
square = square_wave(t)  # [Provided] Generate the target square wave.


# Q2 | Official: increase harmonics, e.g. 20 or 50, and observe accuracy/sharpness.

harmonic_counts = [1, 3, 5, 20, 50]  # [Modified] Extended the provided [1, 3, 5] list.

fig1 = plt.figure(figsize=(12, 8))  # [Provided] Create the reconstruction figure.
plt.subplot(2, 3, 1)  # [Provided] First panel shows the target square wave.
plt.plot(t, square, "k", label="Square wave")  # [Provided] Plot the target signal.
plt.title("Original Square Wave")  # [Provided] Label the target plot.
plt.ylim(-1.5, 1.5)  # [Provided] Keep a consistent amplitude range.
plt.grid(True)  # [Provided] Show a reading grid.
plt.legend()  # [Provided] Show the curve label.

for i, num_harmonics in enumerate(harmonic_counts, start=2):  # [Modified] Plot every selected harmonic count.
    plt.subplot(2, 3, i)  # Select the next panel.
    reconstructed = square_wave_fourier(t, f0, num_harmonics)  # ★ CORE: reconstruct with the chosen harmonic count.
    plt.plot(t, reconstructed, label=f"{num_harmonics} harmonics")  # Plot the approximation.
    plt.plot(t, square, "k--", alpha=0.5, label="Square wave")  # Plot the target for comparison.
    plt.title(f"Fourier Approximation: {num_harmonics} harmonics")  # Label the harmonic count.
    plt.ylim(-1.5, 1.5)  # Keep all panels on the same amplitude scale.
    plt.grid(True)  # Show a reading grid.
    plt.legend()  # Identify approximation and target.

plt.tight_layout()  # Prevent subplot labels from overlapping.
fourier_reconstruction_path = OUTPUT_DIR / "fourier_reconstruction.png"  # [Added] Output path.
save_figure(fig1, fourier_reconstruction_path)  # [Added] Save without opening an unreliable WSL GUI window.


# Q3 | Official: decompose the square wave with the DFT and compare recovered frequencies.

def naive_dft(x):
    """[Provided] Compute the direct DFT using the definition, O(N^2).

    Use: naive_dft(x)
    x: 1D NumPy signal of length N.
    Returns: complex NumPy array X where X[k] is frequency bin k.
    """
    n_samples = len(x)  # Number of samples and frequency bins.
    X = np.zeros(n_samples, dtype=np.complex128)  # Store the complex DFT result.
    for k in range(n_samples):  # k is the output frequency-bin index.
        for n in range(n_samples):  # n is the input sample index.
            angle = -2j * np.pi * k * n / n_samples  # ★ CORE: DFT complex phase -j2πkn/N.
            X[k] += x[n] * np.exp(angle)  # ★ CORE: accumulate sample n's contribution to bin k.
    return X  # Return all frequency-domain coefficients.


benchmark_N = 512  # [Modified] Smaller benchmark size keeps the direct O(N^2) DFT practical.
t_bench = np.linspace(0.0, T, benchmark_N, endpoint=False)  # Create the benchmark time grid.
signal = square_wave_fourier(t_bench, f0, 50)  # ★ CORE: construct the 50-harmonic test signal.
dft_result = naive_dft(signal)  # ★ CORE: convert the time-domain signal to frequency-domain coefficients.

# [Added] Silent correctness check; it produces no output unless the direct DFT is wrong.
assert np.allclose(dft_result, np.fft.fft(signal), atol=1e-8)

frequencies = np.fft.fftfreq(benchmark_N, d=T / benchmark_N)[:benchmark_N // 2]  # [Provided] Positive-frequency axis.
magnitude = 2.0 / benchmark_N * np.abs(dft_result[:benchmark_N // 2])  # [Provided] Positive-frequency magnitude.

fig2 = plt.figure(figsize=(12, 8))  # [Provided] Create the time/frequency comparison figure.
plt.subplot(2, 1, 1)  # Select the time-domain panel.
plt.plot(t_bench, signal)  # Plot the reconstructed square wave.
plt.title("50-Harmonic Square Wave")  # Label the time-domain signal.
plt.xlabel("Time (s)")  # Label the horizontal time axis.
plt.ylabel("Amplitude")  # Label the vertical amplitude axis.
plt.grid(True)  # Show a reading grid.

plt.subplot(2, 1, 2)  # Select the frequency-domain panel.
plt.stem(frequencies, magnitude, basefmt=" ")  # ★ CORE: peaks reveal the recovered harmonic frequencies.
plt.title("DFT Magnitude Spectrum")  # Label the frequency-domain result.
plt.xlabel("Frequency (Hz)")  # Label the frequency axis.
plt.ylabel("Magnitude")  # Label the harmonic-strength axis.
plt.xlim(0, 50)  # Focus on the first 50 Hz.
plt.grid(True)  # Show a reading grid.

for harmonic in range(1, 51, 2):  # [Modified] Extend the provided odd-harmonic guide lines to 50 Hz.
    plt.axvline(harmonic * f0, linestyle="--", alpha=0.35)  # Compare recovered peaks with expected harmonics.

plt.tight_layout()  # Prevent the two panels from overlapping.
dft_spectrum_path = OUTPUT_DIR / "dft_magnitude_spectrum.png"  # [Added] Output path.
save_figure(fig2, dft_spectrum_path)  # [Added] Save the DFT evidence to disk.


# Q4 | Official: modify square_wave, square_wave_fourier and naive_dft using PyTorch operations.

def torch_square_wave(t, f0):
    """[Added] PyTorch version of square_wave.

    Use: torch_square_wave(t, f0)
    t: 1D PyTorch tensor of time samples on CPU or GPU.
    f0: fundamental frequency in Hz.
    Returns: PyTorch tensor containing the square wave on t.device.
    """
    # ★ CORE: same square-wave operation as NumPy, now using PyTorch tensors.
    return torch.sign(torch.sin(2.0 * torch.pi * f0 * t))


def torch_square_wave_fourier(t, f0, num_harmonics):
    """[Added] PyTorch version of the odd-harmonic Fourier reconstruction.

    Use: torch_square_wave_fourier(t, f0, num_harmonics)
    t: 1D PyTorch tensor of time samples.
    f0: fundamental frequency in Hz.
    num_harmonics: number of odd harmonics to add.
    Returns: reconstructed PyTorch signal on t.device.
    """
    result = torch.zeros_like(t)  # Store the accumulated tensor result.
    for k in range(num_harmonics):  # k selects each odd harmonic.
        n = 2 * k + 1  # ★ CORE: generate odd harmonic numbers.
        result += torch.sin(2.0 * torch.pi * n * f0 * t) / n  # ★ CORE: tensor harmonic sum.
    return (4.0 / torch.pi) * result  # Apply the square-wave scale factor.


def torch_naive_dft(x):
    """[Added] Compute the direct DFT with PyTorch tensor operations, O(N^2).

    Use: torch_naive_dft(x)
    x: 1D PyTorch signal on CPU or GPU.
    Returns: complex PyTorch DFT tensor on the same device as x.
    """
    n_samples = x.numel()  # Number of samples and DFT bins.
    n = torch.arange(n_samples, dtype=torch.float64, device=x.device)  # Sample indices n.
    k = n.reshape(-1, 1)  # Frequency indices k as a column vector.
    dft_matrix = torch.exp(-2j * torch.pi * k * n / n_samples)  # ★ CORE: build all exp(-j2πkn/N) basis values.
    X = dft_matrix @ x.to(torch.complex128)  # ★ CORE: matrix multiplication performs every DFT sum.
    return X  # Return the PyTorch frequency-domain coefficients.


t_torch = torch.linspace(0.0, T, N + 1, device=device)[:-1]  # [Added] Match NumPy endpoint=False sampling.
torch_square = torch_square_wave(t_torch, f0)  # Use the PyTorch square-wave function.
torch_fourier = torch_square_wave_fourier(t_torch, f0, 50)  # Use the PyTorch Fourier reconstruction.
torch_dft_test = torch_naive_dft(torch.tensor(signal[:64], dtype=torch.float64))  # Use PyTorch naive DFT on CPU.

# [Added] Silent checks verify that the PyTorch conversions preserve the NumPy results.
assert np.allclose(torch_square.cpu().numpy(), square, atol=1e-5)
assert np.allclose(torch_fourier.cpu().numpy(), square_wave_fourier(t, f0, 50), atol=1e-5)
assert np.allclose(torch_dft_test.numpy(), naive_dft(signal[:64]), atol=1e-8)


# Q5 | Official: create a second naive_dft version that explicitly runs with GPU tensor operations.

def torch_naive_dft_gpu(x):
    """[Added] Run the PyTorch direct DFT explicitly on the CUDA GPU.

    Use: torch_naive_dft_gpu(x)
    x: 1D PyTorch signal; this function moves it to CUDA and complex128.
    Returns: complex PyTorch DFT tensor on cuda:0.
    """
    if not torch.cuda.is_available():  # Stop clearly if CUDA cannot be used.
        raise RuntimeError("CUDA GPU is not available.")

    x_gpu = x.to(device="cuda", dtype=torch.complex128)  # ★ CORE: explicitly place the signal on the GPU.
    return torch_naive_dft(x_gpu)  # ★ CORE: execute the tensor-based O(N^2) DFT on CUDA.


gpu_dft_test = torch_naive_dft_gpu(torch.tensor(signal[:64], dtype=torch.float64))  # ★ CORE: execute direct DFT on CUDA.

# [Added] Silent check verifies the GPU DFT against the NumPy direct DFT.
assert np.allclose(gpu_dft_test.cpu().numpy(), naive_dft(signal[:64]), atol=1e-8)


# Q6 | Official: compare the three computation times and order them fastest to slowest.

gpu_benchmark_signal = torch.tensor(signal, dtype=torch.complex128, device="cuda")  # [Added] Prepare equal GPU input before timing.

start = time.perf_counter()  # [Modified] Start high-resolution CPU timing.
numpy_naive_result = naive_dft(signal)  # Time the NumPy direct DFT.
numpy_naive_time = time.perf_counter() - start  # Record NumPy direct-DFT duration.

torch.cuda.synchronize()  # [Added] Wait for previous CUDA work before starting the GPU timer.
start = time.perf_counter()  # Start GPU timing.
gpu_dft_result = torch_naive_dft_gpu(gpu_benchmark_signal)  # ★ CORE: time the GPU direct DFT.
torch.cuda.synchronize()  # ★ CORE: wait for CUDA completion before stopping the timer.
gpu_naive_time = time.perf_counter() - start  # Record GPU direct-DFT duration.

start = time.perf_counter()  # Start FFT timing.
fft_result = np.fft.fft(signal)  # [Provided] Time NumPy's optimized FFT.
fft_time = time.perf_counter() - start  # Record FFT duration.

# [Added] Silent checks ensure the three timed methods still produce equivalent DFT results.
assert np.allclose(numpy_naive_result, fft_result, atol=1e-8)
assert np.allclose(gpu_dft_result.cpu().numpy(), fft_result, atol=1e-8)

timings = {
    "NumPy FFT": fft_time,
    "PyTorch GPU naive DFT": gpu_naive_time,
    "NumPy naive DFT": numpy_naive_time,
}  # Store measured timings.

ordered_methods = sorted(timings.items(), key=lambda item: item[1])  # ★ CORE: rank methods fastest to slowest.

print("\n--- Q6: Three-Method Performance Comparison ---")  # Print the required timing comparison.
print(f"N = {benchmark_N}")  # Print the benchmark size needed to interpret the result.

for rank, (method, duration) in enumerate(ordered_methods, start=1):  # Print fastest-to-slowest order.
    print(f"{rank}. {method}: {duration:.6f} s")  # Print one ranked timing per method.


# Q7 | Official: change the data size and note timing changes for all three methods.

sample_sizes = [256, 512, 1024, 2048]  # [Added] Four data sizes show the scaling trend.

print("\n--- Q7: Three-Method Scaling Comparison ---")  # Print the required scaling table.
print(f"{'N':>6} {'NumPy naive (s)':>18} {'GPU naive (s)':>16} {'FFT (s)':>12}")  # Print compact headings.

for size in sample_sizes:  # Repeat the same experiment at each input size.
    t_test = np.linspace(0.0, T, size, endpoint=False)  # Create the current time grid.
    signal_test = square_wave_fourier(t_test, f0, 50)  # Create the current 50-harmonic signal.

    start = time.perf_counter()  # Start NumPy direct-DFT timing.
    naive_dft(signal_test)  # Run the NumPy direct DFT.
    numpy_time = time.perf_counter() - start  # Record NumPy direct-DFT time.

    gpu_signal_test = torch.tensor(signal_test, dtype=torch.complex128, device="cuda")  # Prepare equal GPU input.
    torch.cuda.synchronize()  # Wait for previous CUDA work.
    start = time.perf_counter()  # Start GPU timing.
    torch_naive_dft_gpu(gpu_signal_test)  # Run the GPU direct DFT.
    torch.cuda.synchronize()  # Wait until the GPU computation is finished.
    gpu_time = time.perf_counter() - start  # Record GPU direct-DFT time.

    start = time.perf_counter()  # Start FFT timing.
    np.fft.fft(signal_test)  # Run the optimized FFT.
    fft_size_time = time.perf_counter() - start  # Record FFT time.

    print(f"{size:6d} {numpy_time:18.6f} {gpu_time:16.6f} {fft_size_time:12.6f}")  # Print one scaling row.


# Q8 | Official: explain why the observed fastest method is fastest.
# ★ CORE: FFT is O(N log N), while both direct DFT versions remain O(N^2).
# ★ CORE: GPU parallelism accelerates the O(N^2) work but does not change its algorithmic complexity.
# ★ CORE: Small GPU timings can vary because launch/synchronization overhead is large relative to tiny workloads.