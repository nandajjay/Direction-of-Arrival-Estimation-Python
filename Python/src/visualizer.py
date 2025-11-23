import numpy as np
import matplotlib.pyplot as plt
import os

def plot_polar_spectrum(P_avg, angles_rad, out_path):
    """
    Plots the "Red Bulge" (Spatial Spectrum) exactly like your sketch.
    P_avg: The MUSIC spectrum averaged over the duration (or a specific frame).
    """
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    
    # Rotate so 0 degrees is at the top (Standard for Audio)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) # Clockwise
    
    # Close the loop for the plot
    angles_loop = np.append(angles_rad, angles_rad[0])
    P_loop = np.append(P_avg, P_avg[0])
    
    # Plot the "Bulge"
    ax.plot(angles_loop, P_loop, color='red', linewidth=2)
    ax.fill(angles_loop, P_loop, color='red', alpha=0.3)
    
    # Plot the "Black Dot" (Microphone Array Center)
    ax.scatter(0, 0, s=200, c='black', zorder=10, label="Mic Array")
    
    ax.set_title("Spatial Spectrum (Average Direction of Arrival)", fontweight='bold', pad=20)
    ax.set_yticks([]) # Hide radial numbers for cleaner look
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_spatial_spectrogram(P_history, angles_deg, times, out_path):
    """
    Plots Time (y-axis) vs Angle (x-axis) with Color = Probability.
    This shows the history of the 'bulge' moving.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(P_history, aspect='auto', origin='lower', cmap='inferno',
               extent=[angles_deg[0], angles_deg[-1], times[0], times[-1]])
    plt.colorbar(label="MUSIC Spectrum Magnitude")
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Time (Seconds)")
    plt.title("Spatial Spectrogram: Source Movement Over Time")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_waveform_comparison(mic_sig, beam_sig, fs, out_path):
    """
    Compares the raw mic input vs the cleaned beamformed output.
    """
    t = np.arange(len(mic_sig)) / fs
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, mic_sig, color='gray', alpha=0.7)
    plt.title("Raw Microphone Input (Noisy)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, beam_sig, color='blue')
    plt.title("Beamformed Output (Focused)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()