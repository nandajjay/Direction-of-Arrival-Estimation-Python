# run_experiment.py
"""
End-to-end runner script.
Generates simulation, computes GCC & MUSIC, runs Kalman, does beamforming, saves results & plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from simulate import load_source_wave, simulate_moving_source
from gcc_phat import estimate_doa_gcc
from music import estimate_doa_music
from kalman import AngularKalman
from beamformer import delay_and_sum_framewise
from utils import angular_error_deg

def main(outdir="demos", use_real_audio=False, src_path=None):
    os.makedirs(outdir, exist_ok=True)
    fs = 16000
    duration = 4.0
    # load/create source
    if use_real_audio and src_path:
        src, fs = load_source_wave(src_path, duration=duration, fs=fs)
    else:
        src, fs = load_source_wave(None, duration=duration, fs=fs)
    N = len(src)
    t = np.linspace(0, duration, N, endpoint=False)
    # moving angle (true) sweep
    angles_true = np.deg2rad(np.linspace(-60, 60, N))
    # mic geometry (ULA)
    M = 4
    d = 0.05
    mic_positions = np.array([[m*d, 0.0] for m in range(M)])
    # simulate multichannel
    multich = simulate_moving_source(src, angles_true, mic_positions, fs=fs, noise_snr_db=25, frame_len_sec=0.04)
    # save mic wavs
    for m in range(M):
        sf.write(os.path.join(outdir, f"mic_{m+1}.wav"), multich[m], fs)
    # GCC-PHAT estimates
    doa_gcc, n_frames_gcc, win_gcc, hop_gcc = estimate_doa_gcc(multich, mic_positions, fs, win_samps=1024, hop_samps=512, interp=8)
    # MUSIC estimates
    doa_music, n_frames_music, win_music, hop_music = estimate_doa_music(multich, mic_positions, fs, win_samps=1024, hop_samps=512)
    # align frame times to continuous time (center of each frame)
    frame_centers_gcc = np.array([ (i*hop_gcc + win_gcc/2) / fs for i in range(n_frames_gcc) ])
    frame_centers_music = np.array([ (i*hop_music + win_music/2) / fs for i in range(n_frames_music) ])
    true_gcc = np.interp(frame_centers_gcc, t, angles_true)
    true_music = np.interp(frame_centers_music, t, angles_true)
    # Kalman smoothing (use MUSIC if available else GCC)
    dt = hop_music / fs
    kf = AngularKalman(dt=dt, q_angle=1e-5, q_omega=1e-5, r_meas=1e-3)
    angles_kf = []
    for i in range(n_frames_music):
        meas = doa_music[i] if not np.isnan(doa_music[i]) else (doa_gcc[i] if i < len(doa_gcc) else None)
        state = kf.step(meas)
        angles_kf.append(state[0])
    angles_kf = np.array(angles_kf)
    # beamform using tracked angles (music frame rate)
    beamformed = delay_and_sum_framewise(multich, mic_positions, angles_kf, fs, win_music, hop_music)
    sf.write(os.path.join(outdir, "beamformed.wav"), beamformed, fs)
    # plots
    plt.figure(figsize=(10,4))
    plt.plot(frame_centers_gcc, np.rad2deg(true_gcc), label="True (GCC frames)")
    plt.scatter(frame_centers_gcc, np.rad2deg(doa_gcc), s=6, alpha=0.4, label="GCC estimates")
    plt.scatter(frame_centers_music, np.rad2deg(doa_music), s=8, alpha=0.6, label="MUSIC estimates")
    plt.plot(frame_centers_music, np.rad2deg(angles_kf), '-k', linewidth=2, label="Kalman tracked")
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "doa_time_series.png"))
    plt.close()

    # compute basic errors
    valid = ~np.isnan(angles_kf)
    errs = angular_error_deg(angles_kf[valid], true_music[valid])
    mean_err = np.mean(errs) if len(errs)>0 else np.nan
    median_err = np.median(errs) if len(errs)>0 else np.nan
    print(f"Frames used: {len(errs)}, Mean angular error (deg): {mean_err:.2f}, Median: {median_err:.2f}")

    print("Saved outputs to", outdir)
    return True

if __name__ == "__main__":
    main()
