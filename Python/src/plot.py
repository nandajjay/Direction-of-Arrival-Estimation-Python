import matplotlib.pyplot as plt
import numpy as np

def plot_polar_trajectory(angles_rad, times, out_path):
    """
    Generates a polar plot showing the path of the sound source over time.
    - angles_rad: array of estimated angles (radians)
    - times: array of timestamps (for color mapping)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')

    # Set 0 degrees to 'North' (Top) or 'East' (Right). 
    # For standard microphone arrays, 0 is usually broadside (top).
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) # Clockwise
    
    # Filter NaNs
    valid = ~np.isnan(angles_rad)
    th = angles_rad[valid]
    t = times[valid]
    
    # Use radius to indicate time or confidence. Here we use constant radius 
    # but change color to show time evolution (dark -> bright).
    # We add a slight radial jitter so lines don't perfectly overlap if stationary.
    r = np.ones_like(th) + np.linspace(0, 0.1, len(th))
    
    # Scatter plot with colormap to show time progression
    sc = ax.scatter(th, r, c=t, cmap='viridis', alpha=0.7, s=20, label='Estimated Path')
    
    # Add arrow for the FINAL estimated angle
    if len(th) > 0:
        final_angle = th[-1]
        ax.annotate('Final Position', xy=(final_angle, 1.1), xytext=(final_angle, 1.4),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    ha='center', color='red', fontweight='bold')

    # Aesthetics
    ax.set_rmax(1.2)
    ax.set_rticks([])  # Hide radial ticks (less clutter)
    ax.set_title("Source Trajectory (Color = Time)", pad=20, fontweight='bold')
    
    # Colorbar to show time scale
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Time (s)')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved polar trajectory to {out_path}")

# --- Usage in main() ---
# Inside run_experiment.py, after the Kalman loop:
# plot_polar_trajectory(angles_kf, frame_centers_music, os.path.join(outdir, "doa_polar.png"))