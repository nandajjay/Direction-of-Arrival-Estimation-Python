# beamformer.py
"""
Delay-and-sum beamformer using fractional delays (FFT phase shift).
Functions:
- fractional_delay_fft (reused)
- delay_and_sum_beamform: frame-based fractional-delay delay-and-sum using tracked angles
"""

import numpy as np
from numpy.fft import fft, ifft

def fractional_delay_fft(signal, tau_sec, fs):
    N = len(signal)
    Nfft = 1 << (N-1).bit_length()
    X = fft(signal, Nfft)
    freqs = np.fft.fftfreq(Nfft, 1.0/fs)
    phase = np.exp( -2j * np.pi * freqs * tau_sec )
    y = np.real(ifft(X * phase, Nfft))[:N]
    return y

def delay_and_sum_framewise(multich, mic_positions, angles_rad_per_frame, fs, win_samps, hop_samps):
    """
    Build beamformed signal by per-frame fractional-delay alignment based on angles (per-frame).
    - multich: (M, N)
    - angles_rad_per_frame: (n_frames,) tracked angle per frame (radians)
    - returns: beamformed 1D array length N
    """
    M, N = multich.shape
    n_frames = len(angles_rad_per_frame)
    out = np.zeros(N)
    win = np.hanning(win_samps)
    c = 343.0
    for f in range(n_frames):
        start = f * hop_samps
        end = min(N, start + win_samps)
        if end - start < win_samps:
            # pad short segment
            seg_len = end - start
            segments = [np.zeros(win_samps) for _ in range(M)]
            for m in range(M):
                seg = multich[m, start:end]
                segments[m][:seg_len] = seg * win[:seg_len]
        else:
            segments = [multich[m, start:end] * win for m in range(M)]
        theta = angles_rad_per_frame[f]
        u = np.array([np.cos(theta), np.sin(theta)])
        taus = - (mic_positions @ u) / c
        aligned = np.zeros(win_samps)
        for m in range(M):
            delayed = fractional_delay_fft(segments[m], -taus[m], fs)  # note sign so source aligns
            aligned += delayed
        out[start:end] += aligned[:end-start]
    maxv = np.max(np.abs(out)) + 1e-12
    out = out / maxv * 0.98
    return out
