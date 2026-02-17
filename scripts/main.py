import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ==============================
# Setup Paths
# ==============================
# The script will now look for these dictionaries relative to itself
dict_dir = "dictionaries"
output_dir = "data/processed/nmr_images"
os.makedirs(output_dir, exist_ok=True)

# Try to load Dictionaries
try:
    ald_df = pd.read_excel(os.path.join(dict_dir, "aldehyde_dictionary.xlsx"))
    am_df = pd.read_excel(os.path.join(dict_dir, "amine_dictionary.xlsx"))
    ald_map = dict(zip(ald_df['Number'], ald_df['Name']))
    am_map = dict(zip(am_df['Number'], am_df['Name']))
except Exception as e:
    print(f"Error loading dictionaries: {e}")
    # Fallback placeholders so the script doesn't crash
    ald_map, am_map = {}, {}

# ==============================
# Processing Functions
# ==============================
def next_power_of_2(x):
    return 1 << (x - 1).bit_length()

def baseline_als(y, lam=1e6, p=0.001, niter=10):
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
# Search and Process Loop
# ==============================
print("Searching for NMR data folders...")
found_data = False

# We walk through the entire current directory and subdirectories
for root, dirs, files in os.walk("."):
    for d in dirs:
        if d == "PRESAT_01.fid":
            found_data = True
            actual_fid_path = os.path.join(root, d)
            
            # The parent folder (e.g., 1_ald_1_am.fid) contains the IDs
            parent_folder = os.path.basename(root)
            
            # Extract IDs using regex
            match = re.search(r'(\d+)_ald_(\d+)_am', parent_folder, re.IGNORECASE)
            
            if match:
                ald_id = int(match.group(1))
                am_id = int(match.group(2))
                
                aldehyde_name = ald_map.get(ald_id, f"Aldehyde_{ald_id}")
                amine_name = am_map.get(am_id, f"Amine_{am_id}")
                
                plot_filename = f"{amine_name.replace(' ', '_')}_{aldehyde_name.replace(' ', '_')}.png"
                
                print(f"Found match: {parent_folder} -> {plot_filename}")
                
                try:
                    # NMR Processing
                    dic, data = ng.varian.read(actual_fid_path)
                    data = np.asarray(data).ravel()
                    
                    zf_size = next_power_of_2(data.size * 2)
                    data_zf = ng.proc_base.zf_size(data, zf_size)
                    spec = ng.proc_base.fft(data_zf)
                    spec /= spec.size
                    spec = ng.proc_base.ps(spec, p0=0.0, p1=0.0)
                    spec_real = np.real(spec).ravel()

                    spec_plot = spec_real[:30000]
                    baseline = baseline_als(spec_plot, lam=1e6, p=0.001)
                    spec_corrected = spec_plot - baseline

                    # Plotting
                    plt.figure(figsize=(15, 8))
                    plt.plot(spec_corrected, linewidth=0.8, color='black')
                    plt.xlim(0, 30000)
                    ymax = np.max(np.abs(spec_corrected))
                    plt.ylim(-0.05 * ymax, 1.1 * ymax)
                    plt.title(f"{amine_name} and {aldehyde_name} NMR")
                    
                    # Save
                    outfile = os.path.join(output_dir, plot_filename)
                    plt.savefig(outfile, dpi=300)
                    plt.close()
                    print(f"Successfully saved: {outfile}")
                    
                except Exception as e:
                    print(f"Error processing {actual_fid_path}: {e}")
            else:
                print(f"Found PRESAT_01.fid in {root}, but parent folder name didn't match ID pattern.")

if not found_data:
    print("No folders named 'PRESAT_01.fid' were found in the current directory or its subfolders.")