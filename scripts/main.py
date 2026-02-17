import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ==============================
# Paths
# ==============================
base_dir = "/nobackup/xtst45/NMRGlue_test/data/raw/testdata"
output_dir = "/nobackup/xtst45/NMRGlue_test/data/processed"
os.makedirs(output_dir, exist_ok=True)

fid_folders = [
    os.path.join(base_dir, "PRESAT_01.fid"),
    os.path.join(base_dir, "scoutfids", "PRESAT_01_Scout1D.fid"),
]


# ==============================
# Helper functions
# ==============================
def next_power_of_2(x):
    return 1 << (x - 1).bit_length()


def baseline_als(y, lam=1e6, p=0.001, niter=10):
    """
    Asymmetric Least Squares baseline correction
    Closest equivalent to MestReNova baseline correction
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


# ==============================
# Processing loop
# ==============================
for fid_dir in fid_folders:
    plot_name = os.path.basename(fid_dir).replace(".fid", "")

    # Read Varian FID
    dic, data = ng.varian.read(fid_dir)
    data = np.asarray(data).ravel()

    # Zero filling (2×, next power of 2)
    zf_size = next_power_of_2(data.size * 2)
    data_zf = ng.proc_base.zf_size(data, zf_size)

    # Fourier transform
    spec = ng.proc_base.fft(data_zf)

    # Normalize intensity
    spec /= spec.size

    # Phase correction (adjust p0 if needed)
    spec = ng.proc_base.ps(spec, p0=0.0, p1=0.0)

    # Real spectrum
    spec_real = np.real(spec).ravel()

    # Limit to first 20000 points
    spec_plot = spec_real[:30000]

    # Baseline correction (ALS – no ringing)
    baseline = baseline_als(spec_plot, lam=1e6, p=0.001)
    spec_corrected = spec_plot - baseline

    # ==============================
    # Plot
    # ==============================
    plt.figure(figsize=(30, 18))
    plt.plot(spec_corrected, linewidth=0.8)

    plt.xlim(0, 30000)

    ymax = np.max(np.abs(spec_corrected))
    plt.ylim(-0.05 * ymax, 1.1 * ymax)

    plt.xlabel("Points")
    plt.ylabel("Intensity")
    plt.title(f"NMR Spectrum: {plot_name}")
    plt.tight_layout()

    # Save
    outfile = os.path.join(output_dir, f"{plot_name}.png")
    plt.savefig(outfile, dpi=300)
    plt.close()

    print(f"Saved plot: {outfile}")
