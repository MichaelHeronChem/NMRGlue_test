import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
import glob
import re

# --- Configuration ---
BASE_DIR = os.path.abspath("/nobackup/xtst45/NMRGlue_test")
RAW_DATA_ROOT = os.path.join(BASE_DIR, "data/raw")
DICT_DIR = os.path.join(BASE_DIR, "dictionaries")
ALDEHYDE_DICT = os.path.join(DICT_DIR, "aldehyde_dictionary.xlsx")
AMINE_DICT = os.path.join(DICT_DIR, "amine_dictionary.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/nmr_images")

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
    """Robust lookup: finds the first two numbers in the folder name."""
    try:
        # Uses regex to find all numbers in the folder name
        numbers = re.findall(r"\d+", folder_name)
        if len(numbers) < 2:
            return None, None

        ald_id, am_id = int(numbers[0]), int(numbers[1])

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
    except Exception:
        return None, None


def process_spectrum(fid_path, am_nm, ald_nm, save_path):
    # 1. Read and FFT
    dic, data = ng.varian.read(fid_path)
    data = np.asarray(data).ravel()
    zf_size = next_power_of_2(data.size * 2)
    data_zf = ng.proc_base.zf_size(data, zf_size)
    spec = ng.proc_base.fft(data_zf)
    spec /= spec.size
    spec_real = np.real(spec).ravel()

    # 2. Baseline
    spec_corrected = spec_real - baseline_als(spec_real)

    # 3. PPM Scale Calibration
    obs_mhz = float(dic["procpar"]["sreffrq"]["values"][0])
    if obs_mhz < 300:
        obs_mhz = 600.0
    sw_hz = float(dic["procpar"]["sw"]["values"][0])

    freq_hz = np.linspace(sw_hz / 2, -sw_hz / 2, zf_size)
    ppm_scale = freq_hz / obs_mhz

    # 4. Reference and Suppression
    max_idx = np.argmax(spec_corrected)
    ppm_scale += 1.98 - ppm_scale[max_idx]

    # Gaussian Mask
    sigma = 0.5 / (2 * np.sqrt(2 * np.log(2)))
    suppression_mask = 1 - (1 - 0.005) * np.exp(
        -((ppm_scale - 1.98) ** 2) / (2 * sigma**2)
    )
    spec_corrected *= suppression_mask

    # 5. Peak Picking
    peaks, _ = find_peaks(
        spec_corrected, height=0.003 * np.max(spec_corrected), distance=zf_size // 200
    )

    # 6. Plotting
    plt.figure(figsize=(18, 10))
    plt.plot(ppm_scale, spec_corrected, color="black", linewidth=0.7)

    ymax = np.max(spec_corrected)
    for p in peaks:
        if -0.5 <= ppm_scale[p] <= 14 and abs(ppm_scale[p] - 1.98) > 0.15:
            plt.text(
                ppm_scale[p],
                spec_corrected[p] + (0.01 * ymax),
                f"{ppm_scale[p]:.2f}",
                rotation=90,
                ha="center",
                fontsize=8,
                color="darkred",
                weight="bold",
            )

    plt.gca().invert_xaxis()
    plt.xlim(14, -0.5)

    mask = (ppm_scale <= 14) & (ppm_scale >= -0.5)
    plt.ylim(-0.05 * np.max(spec_corrected[mask]), 1.3 * np.max(spec_corrected[mask]))

    plt.title(f"NMR Analysis: {am_nm} + {ald_nm}", fontsize=16)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# --- Main Loop ---
fid_search_pattern = os.path.join(RAW_DATA_ROOT, "**", "*.fid")
fid_folders = glob.glob(fid_search_pattern, recursive=True)

stats = {"success": 0, "skipped_scout": 0, "failed_lookup": 0, "errors": 0}

for fid_path in fid_folders:
    if "scout" in fid_path.lower():
        stats["skipped_scout"] += 1
        continue

    parent_folder = os.path.basename(os.path.dirname(fid_path))
    am_nm, ald_nm = get_compound_names(parent_folder, ALDEHYDE_DICT, AMINE_DICT)

    if am_nm is None:
        print(f"Skipping: Metadata lookup failed for folder '{parent_folder}'")
        stats["failed_lookup"] += 1
        continue

    fid_subname = os.path.basename(fid_path).replace(".fid", "")
    save_path = os.path.join(OUTPUT_DIR, f"{am_nm}_{ald_nm}_{fid_subname}.png")

    try:
        process_spectrum(fid_path, am_nm, ald_nm, save_path)
        stats["success"] += 1
    except Exception as e:
        print(f"Error processing {fid_path}: {e}")
        stats["errors"] += 1

print(f"""
--- Batch Process Complete ---
Successfully Saved: {stats["success"]}
Skipped (Scout):    {stats["skipped_scout"]}
Skipped (Unknown):  {stats["failed_lookup"]}
Errors:             {stats["errors"]}
------------------------------
""")
