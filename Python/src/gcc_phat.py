# gcc_phat.py
"""
GCC-PHAT based TDOA and DOA helper functions.

Functions:
- gcc_phat: compute TDOA between two signals using PHAT weighting and zero-padding for subsample resolution
- tdoa_to_angle_pairwise: convert pairwise TDOA and mic pair geometry to angle (far-field)
- estimate_doa_gcc: frame-based wrapper to compute angle per frame using multiple mic pairs
"""

import numpy as np
from numpy.fft import fft, ifft
import math

def gcc_phat(sig1, sig2, fs, max_tau=None, interp=8):
    """
    Cross-correlation via GCC-PHAT with interpolation.
    - interp controls zero-padding factor for sub-sample resolution.
    Returns tau (seconds) and cross-correlation array (if needed)
    """
    n = len(sig1) + len(sig2)
    N = 1 << (n-1).bit_length()
    N *= interp
    SIG1 = fft(sig1, N)
    SIG2 = fft(sig2, N)
    R = SIG1 * np.conj(SIG2)
    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    R /= denom
    cc = np.real(ifft(R, N))
    # Rearrange cross-correlation to -N/2..N/2
    cc = np.concatenate((cc[-(N//2):], cc[:N//2]))
    max_shift = int((N/2) if max_tau is None else min(N/2, np.ceil(fs*max_tau*interp)))
    shift = np.argmax(np.abs(cc)) - N//2
    tau = shift / float(fs * interp)
    return tau, cc

def tdoa_to_angle_pairwise(tau, mic_pos_i, mic_pos_j, sound_speed=343.0):
    """
    Convert TDOA between mic i and j to far-field DOA angle (radians).
    Returns angle in [-pi, pi] or None if invalid (|val|>1).
    """
    d = mic_pos_j - mic_pos_i
    dist = np.linalg.norm(d)
    if dist < 1e-8:
        return None
    phi = math.atan2(d[1], d[0])
    val = (sound_speed * tau) / dist
    if abs(val) > 1.0:
        return None
    # choose principal solution that maps to -pi..pi
    ang = phi - math.acos(val)
    ang = (ang + np.pi) % (2*np.pi) - np.pi
    return ang

def frame_signals(x, win_samps, hop_samps, window=None):
    """
    Return array of frames (num_frames x win_samps) with hann window applied if provided.
    """
    N = len(x)
    frames = int(np.ceil((N - win_samps) / hop_samps)) + 1
    out = np.zeros((frames, win_samps))
    for i in range(frames):
        start = i * hop_samps
        end = start + win_samps
        seg = np.zeros(win_samps)
        if start < N:
            seg[:max(0, min(N-start, win_samps))] = x[start:min(end, N)]
        if window is not None:
            seg = seg * window
        out[i] = seg
    return out

def estimate_doa_gcc(multich, mic_positions, fs, win_samps=512, hop_samps=256, interp=8):
    """
    Estimate per-frame DOA using GCC-PHAT across all mic pairs.
    Returns:
    - doa_angles_rad (num_frames,) (NaN where estimation failed)
    """
    M, N = multich.shape
    window = np.hanning(win_samps)
    frames_all = [frame_signals(multich[m], win_samps, hop_samps, window) for m in range(M)]
    n_frames = frames_all[0].shape[0]
    mic_pairs = [(i,j) for i in range(M) for j in range(i+1, M)]
    doa_est = np.full(n_frames, np.nan)
    for f in range(n_frames):
        thetas = []
        for (i,j) in mic_pairs:
            x = frames_all[i][f]
            y = frames_all[j][f]
            tau, _ = gcc_phat(x, y, fs, interp=interp)
            th = tdoa_to_angle_pairwise(tau, mic_positions[i], mic_positions[j])
            if th is not None:
                thetas.append(th)
        if thetas:
            # circular mean
            s = np.mean(np.sin(thetas))
            c = np.mean(np.cos(thetas))
            doa_est[f] = math.atan2(s, c)
    return doa_est, n_frames, win_samps, hop_samps
