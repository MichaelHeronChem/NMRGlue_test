import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks

# --- Configuration ---
BASE_DIR = os.path.abspath("/nobackup/xtst45/NMRGlue_test")
DATA_PATH = os.path.join(BASE_DIR, "data/raw/1_ald_3_am/PRESAT_01.fid")
DICT_DIR = os.path.join(BASE_DIR, "dictionaries")
ALDEHYDE_DICT = os.path.join(DICT_DIR, "aldehyde_dictionary.xlsx")
AMINE_DICT = os.path.join(DICT_DIR, "amine_dictionary.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/nmr_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Helper Functions ---
def next_power_of_2(x):
    return 1 << (x - 1).bit_length()


def baseline_als(y, lam=1e6, p=0.001, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * (D @ D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def get_compound_names(folder_name, ald_path, am_path):
    try:
        parts = folder_name.split("_")
        ald_id, am_id = int(parts[0]), int(parts[2])
        df_ald = pd.read_excel(ald_path)
        df_am = pd.read_excel(am_path)
        ald_name = (
            str(df_ald.loc[df_ald["Number"] == ald_id, "Name"].values[0])
            .strip()
            .replace(" ", "_")
        )
        am_name = (
            str(df_am.loc[df_am["Number"] == am_id, "Name"].values[0])
            .strip()
            .replace(" ", "_")
        )
        return am_name, ald_name
    except:
        return "UnknownAmine", "UnknownAldehyde"


# --- Main Execution ---
parent_folder = os.path.basename(os.path.dirname(DATA_PATH))
am_nm, ald_nm = get_compound_names(parent_folder, ALDEHYDE_DICT, AMINE_DICT)
full_save_path = os.path.join(OUTPUT_DIR, f"{am_nm}_{ald_nm}.png")

dic, data = ng.varian.read(DATA_PATH)
data = np.asarray(data).ravel()

# Process FID
zf_size = next_power_of_2(data.size * 2)
data_zf = ng.proc_base.zf_size(data, zf_size)
spec = ng.proc_base.fft(data_zf)
spec /= spec.size
spec_real = np.real(spec).ravel()

# Baseline correction
spec_corrected = spec_real - baseline_als(spec_real)

# --- Calibration ---
obs_mhz = float(dic["procpar"]["sreffrq"]["values"][0])
if obs_mhz < 300:
    obs_mhz = 600.0
sw_hz = float(dic["procpar"]["sw"]["values"][0])

freq_hz = np.linspace(sw_hz / 2, -sw_hz / 2, zf_size)
ppm_scale = freq_hz / obs_mhz

# Step 1: Reference to 1.98 (Centering)
max_idx = np.argmax(spec_corrected)
offset = 1.98 - ppm_scale[max_idx]
ppm_scale += offset

# --- Step 2: Gaussian Solvent Suppression ---
# Instead of zeroing, we multiply the region by an inverted Gaussian
center = 1.98
fwhm = 0.5  # Width of suppression in ppm
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

# Calculate the suppression mask (1.0 everywhere, dipping to near 0 at 1.98)
# We use a depth factor (e.g., 0.01) to ensure the peak is heavily attenuated
suppression_depth = 0.005
gaussian_bell = np.exp(-((ppm_scale - center) ** 2) / (2 * sigma**2))
suppression_mask = 1 - (1 - suppression_depth) * gaussian_bell

spec_corrected = spec_corrected * suppression_mask

# --- Peak Picking ---
height_thresh = 0.03 * np.max(spec_corrected)
peaks, _ = find_peaks(spec_corrected, height=height_thresh, distance=zf_size // 200)

# --- Plotting ---
plt.figure(figsize=(18, 10))
plt.plot(ppm_scale, spec_corrected, color="black", linewidth=0.7)

# Label remaining peaks
ymax_global = np.max(spec_corrected)
for p in peaks:
    peak_ppm = ppm_scale[p]
    peak_height = spec_corrected[p]
    if -0.5 <= peak_ppm <= 14:
        # Ignore labeling anything very close to the solvent center
        if abs(peak_ppm - 1.98) > 0.1:
            plt.text(
                peak_ppm,
                peak_height + (0.01 * ymax_global),
                f"{peak_ppm:.2f}",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
                color="darkred",
                weight="bold",
            )

plt.gca().invert_xaxis()
plt.xlim(14, -0.5)

# Scaling mask
mask_plot = (ppm_scale <= 14) & (ppm_scale >= -0.5)
if any(mask_plot):
    ymax_visible = np.max(spec_corrected[mask_plot])
    plt.ylim(-0.05 * ymax_visible, 1.3 * ymax_visible)

plt.xlabel(r"Chemical Shift ($\delta$, ppm)", fontsize=14)
plt.ylabel("Intensity", fontsize=14)
plt.title(
    f"NMR Analysis (Gaussian Solvent Suppression): {am_nm} + {ald_nm}", fontsize=16
)
plt.grid(True, alpha=0.1, linestyle="--")

plt.savefig(full_save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Success! Final image saved to: {full_save_path}")
