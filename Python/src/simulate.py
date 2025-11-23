# simulate.py
"""
Microphone array simulator.

Functions:
- simulate_moving_source: create multichannel WAV by applying fractional delays to a
  single-channel source signal according to a moving far-field source angle.
- load_source_wave: convenience to load a wav or synthesize a chirp if no file provided.

Outputs:
- multichannel numpy array (M x N)
- true_angles_rad (N)
"""

import numpy as np
import soundfile as sf
from scipy.signal import chirp, get_window
from numpy.fft import fft, ifft

def load_source_wave(path=None, duration=4.0, fs=16000):
    """
    Load a source wave from file or create a synthetic chirp if path is None.
    Returns (signal, fs)
    """
    if path:
        sig, sr = sf.read(path)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        if sr != fs:
            # simple resample using librosa (import only when needed)
            try:
                import librosa
                sig = librosa.resample(sig.astype(float), sr, fs)
            except Exception:
                raise RuntimeError("Install librosa to resample or supply a file at the target fs")
        return sig.astype(np.float32), fs
    else:
        t = np.linspace(0, duration, int(fs*duration), endpoint=False)
        s = 0.6 * chirp(t, f0=300, f1=2000, t1=duration, method='linear')
        s += 0.4 * 0.3 * np.sin(2*np.pi*220*t)
        s = s / (np.max(np.abs(s)) + 1e-12) * 0.95
        return s.astype(np.float32), fs

def fractional_delay_fft(signal, tau_sec, fs):
    """
    Apply a fractional delay tau_sec to a 1D signal using frequency-domain phase shift.
    This assumes periodic extension; for long signals, window or overlap-add can be used.
    """
    N = len(signal)
    # Zero-pad to next power-of-two for better resolution and to avoid circular artifacts
    Nfft = 1 << (N-1).bit_length()
    X = fft(signal, Nfft)
    freqs = np.fft.fftfreq(Nfft, 1.0/fs)
    phase = np.exp(-2j * np.pi * freqs * tau_sec)
    y = np.real(ifft(X * phase, Nfft))[:N]
    return y

def simulate_moving_source(src, angles_rad, mic_positions, fs=16000, noise_snr_db=40, frame_len_sec=0.05):
    """
    Simulate multichannel recordings of a far-field moving source.
    - src: 1D numpy (N)
    - angles_rad: per-sample true angle array (N) in radians
    - mic_positions: (M,2) array of (x,y) coordinates in meters
    - frame_len_sec: length of frames to apply approx-constant delay
    Returns:
    - multich: (M, N) numpy array
    - true_angles_rad: (N,)
    """
    c = 343.0
    N = src.shape[0]
    M = mic_positions.shape[0]
    multich = np.zeros((M, N), dtype=np.float32)
    frame_len = int(round(frame_len_sec*fs))
    # process frame-by-frame for moving source
    for start in range(0, N, frame_len):
        end = min(N, start + frame_len)
        # take mid-frame angle
        mid_idx = min(end-1, start + (end-start)//2)
        theta = angles_rad[mid_idx]
        u = np.array([np.cos(theta), np.sin(theta)])
        taus = - (mic_positions @ u) / c  # seconds
        for m in range(M):
            delayed = fractional_delay_fft(src[start:end], taus[m], fs)
            multich[m, start:end] += delayed
    # add white Gaussian noise to reach specified SNR (approx)
    sig_power = np.mean(src**2)
    noise_power = sig_power / (10**(noise_snr_db/10.0))
    noise = np.sqrt(noise_power) * np.random.randn(*multich.shape)
    multich += noise
    # normalize to avoid clipping
    maxv = np.max(np.abs(multich)) + 1e-12
    multich = multich / maxv * 0.98
    return multich

if __name__ == "__main__":
    # quick sanity test (creates demo files)
    import os
    fs = 16000
    src, fs = load_source_wave("test_audio1.wav", duration=4.0, fs=fs)
    t = np.linspace(0, 4.0, src.size, endpoint=False)
    angles = np.deg2rad(np.linspace(-60, 60, src.size))
    mic_positions = np.array([[0.0,0.0],[0.05,0.0],[0.10,0.0],[0.15,0.0]])
    multich = simulate_moving_source(src, angles, mic_positions, fs=fs, noise_snr_db=30)
    os.makedirs("demos", exist_ok=True)
    import soundfile as sf
    for m in range(multich.shape[0]):
        sf.write(f"demos/mic_{m+1}.wav", multich[m], fs)
    print("Saved demo mic wavs to ./demos/")
