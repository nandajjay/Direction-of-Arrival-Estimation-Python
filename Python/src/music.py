# music.py
"""
Broadband MUSIC implementation (frequency-bin averaging).

Functions:
- steering_vector_far_field: construct steering vector for given angle and frequency bin
- music_pseudospectrum_broadband: compute MUSIC pseudospectrum averaged across selected freq bins
- estimate_doa_music: per-frame wrapper that returns peak angle per frame
"""

import numpy as np
from numpy.linalg import eigh
import math

def steering_vector_far_field(theta_rad, mic_positions, freq, sound_speed=343.0):
    """
    Steering vector for a plane wave at angle theta (radians) for a given freq (Hz).
    mic_positions: (M,2)
    returns: (M,) complex
    """
    k = 2 * np.pi * freq / sound_speed
    u = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    phase = np.exp(-1j * k * (mic_positions @ u))
    return phase

def music_pseudospectrum_broadband(snapshot_matrix, mic_positions, fs, n_fft=1024, freqs=None, angles_deg=np.arange(-90,91,1), noise_subspace_dim= None):
    """
    snapshot_matrix: (M, win_samps) time snapshots for frame
    freqs: list/array of frequencies (Hz) to average across (if None, pick a range e.g., 300-3000Hz)
    returns: angles (deg), pseudospectrum (len(angles),)
    """
    M, L = snapshot_matrix.shape
    # compute sample covariance (M x M)
    R = np.cov(snapshot_matrix)
    # eigen-decomposition
    vals, vecs = eigh(R)
    idx = np.argsort(np.abs(vals))[::-1]
    if noise_subspace_dim is None:
        # assume 1 source => noise subspace dim = M-1
        noise_subspace_dim = max(1, M-1)
    En = vecs[:, idx[noise_subspace_dim:]]  # noise eigenvectors
    if freqs is None:
        freqs = np.linspace(300, min(4000, fs/2*0.9), 8)
    angles_rad = np.deg2rad(angles_deg)
    P = np.zeros(len(angles_rad))
    for fi in freqs:
        for ai, th in enumerate(angles_rad):
            a = steering_vector_far_field(th, mic_positions, fi)
            denom = np.conj(a).T @ (En @ En.conj().T) @ a
            P[ai] += 1.0 / (np.abs(denom) + 1e-12)
    P = 10 * np.log10(np.real(P) + 1e-12)
    return angles_deg, P

def estimate_doa_music(multich, mic_positions, fs, win_samps=512, hop_samps=256, angles_deg=np.arange(-90,91,1)):
    """
    Per-frame MUSIC estimation. Returns doa_estimates (n_frames,) in radians (NaN for failure)
    """
    M, N = multich.shape
    window = np.hanning(win_samps)
    frames_all = [None]*M
    for m in range(M):
        # simple framing
        frames_all[m] = []
        # generate frames
        for start in range(0, N - win_samps + 1, hop_samps):
            seg = multich[m, start:start+win_samps] * window
            frames_all[m].append(seg)
        frames_all[m] = np.stack(frames_all[m], axis=0)  # (n_frames, win_samps)
    n_frames = frames_all[0].shape[0]
    doa_est = np.full(n_frames, np.nan)
    for f in range(n_frames):
        snapshot = np.array([frames_all[m][f] for m in range(M)])  # (M, win_samps)
        _, P = music_pseudospectrum_broadband(snapshot, mic_positions, fs, freqs=None, angles_deg=angles_deg)
        peak_idx = np.argmax(P)
        doa_est[f] = np.deg2rad(angles_deg[peak_idx])
    return doa_est, n_frames, win_samps, hop_samps
