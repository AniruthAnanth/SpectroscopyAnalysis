# Import necessary libraries for parallel processing, plotting, file handling, and mathematical operations.
from mpi4py import MPI
import matplotlib.pyplot as plt
import time
import LightweightTransferMatrixMethod
import os
import numpy as np
import pandas as pd
from ri import RefractiveIndexMaterial
import random
import json
import copy

# Define constants for the number of wavelengths, precision, and sample thickness.
NUM_WAVELENGTHS = 25
DEFINITION = 100
DROPLET_DEPTH = 200
NUM_SAMPLES = 1200 # Set the number of samples
SAVE_DATA = True
VERBOSE = 2

# Seed the random number generator for reproducibility.
#random.seed(42)

# Initialize MPI communication and get the rank and size of the current process.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Capture the start time of the program.
program_start = MPI.Wtime()

# Function to save refractive index data for a material at specified wavelengths to a CSV file.
def save(material, wavelengths, filename):
    out_str = "wavelength,n,kappa"
    
    # Loop through each wavelength, get n and kappa, and append to the output string.
    for wavelength in wavelengths:
        n, kappa = get_n_kappa(material, wavelength)
        out_str += f"\n{wavelength},{n},{kappa}"
    
    # Write the output string to the specified file.
    f_out = open(out_folder + filename, "w")
    f_out.write(out_str)
    f_out.close()

# Function to retrieve refractive index (n) and extinction coefficient (kappa) for a material at a wavelength.
def get_n_kappa(material, wavelength):
    n = 0
    kappa = 0
    
    # Get refractive index and attempt to get the extinction coefficient.
    n = material.get_refractive_index(wavelength)
    try:
        kappa = material.get_extinction_coefficient(wavelength)
    except:
        kappa = 0
    return max(n, 0), max(kappa, 0)

# Function to find a VOC (volatile organic compound) by name in the database paths and return its metadata.
def find_voc_in_db(voc_name):
    organic_path = database_path + "data-nk/organic/"
    main_path = database_path + "data-nk/main/"
    
    organic_filenames = os.listdir(organic_path)
    main_filenames = os.listdir(main_path)
    voc_filename = None
    voc_name.lower()
    
    # Check organic directory for the VOC file by matching names.
    for filename in organic_filenames:
        if voc_name in filename.split(' - '):
            voc_filename = organic_path + filename
            break
    
    # If not found, check the main directory.
    if not voc_filename:
        for filename in main_filenames:
            if voc_name in filename.split(' - '):
                voc_filename = main_path + filename
                break

    # Raise an exception if the VOC file is still not found.
    if not voc_filename:
        raise Exception(
            f"{voc_name} does not exist. All possible organic compounds listed below: \n{organic_filenames}")
    
    # Get the file page names for the VOC.
    voc_page_names = os.listdir(voc_filename)
    voc_filename, voc_shelf = voc_filename.split('/')[-1], voc_filename.split('/')[-2]
    
    # Remove file extensions for easier access.
    for i in range(len(voc_page_names)):
        voc_page_names[i] = voc_page_names[i].replace('.yml', '')

    return voc_shelf, voc_filename, voc_page_names

# Function to generate refractive index from a CSV file.
def generate_refractive_index_from_csv(csv):
    df = pd.read_csv(csv)
    lbdas = df['wavelength'][1:].to_numpy().astype(np.float64)[0:]
    n_s = df['n'][1:].to_numpy().astype(np.float64)[0:]
    k_s = df['kappa'][1:].to_numpy().astype(np.float64)[0:]

    # Function to retrieve data points for a specific wavelength.
    def raw_data_points(lbda):
        i = df[df['wavelength'] == lbda].index[0]
        return n_s[i] - 1j * k_s[i]

    raw_data_points.__name__ = csv + " refractive index"

    return raw_data_points

# Define output folder and database paths.
out_folder = "./csv/"
database_path = "../refractiveindex.info-database/database/"
wavelengths = np.linspace(529, 585, NUM_WAVELENGTHS + 1)
vocs = [
    #["C2H4", 0, 0.05, None],                 # Ethylene # 0.00005
    #["C2H4O2", 0, 0.01, None],               # Acetic acid # 0.00001
    #["C2H6", 0, 0.01, None],                 # Ethane # 0.00001
    ["C2H6O", 1, 0.06, None],                # Ethanol # 0.00006
    #["C3H6O", 1, 0.07, None],                # Acetone # 0.00007
    ["C3H8O", 1, 0.02, None],                # Propanol # 0.00002
    #["C4H8O2", 1, 0.03, None],               # Ethyl acetate # 0.00003
    ["C6H6", 1, 0.02, None],                 # Benzene # 0.00002
    ["C7H8", 1, 0.06, None],                 # Toluene # 0.00006
    #["C8H10", 0, 0.03, None],                # Xylene # 0.00003
    ["CH4", 0, 0.17, None],                  # Methane # 0.00017
    ["CH4O", 0, 0.05, None],                 # Methanol # 0.00005
]

# If the current process is the root, load VOC refractive indices and save to files.
if rank == 0:
    if VERBOSE > 0: print("Loading VOC refractive indices...")
    for i in range(len(vocs)):
        voc_shelf, voc_book_name, voc_page_names = find_voc_in_db(vocs[i][0])
        voc_material = RefractiveIndexMaterial(shelf=voc_shelf, book=voc_book_name, page=voc_page_names[vocs[i][1]])
        save(voc_material, wavelengths, voc_book_name + ".csv")
        vocs[i][3] = out_folder + voc_book_name + ".csv"
        if VERBOSE > 1: print(f"Loaded {voc_book_name}...")
    if VERBOSE > 0: print("Loaded all VOC's!")

# Broadcast the VOC data to all processes.
vocs = comm.bcast(vocs, root=0)

# Function to generate a batch of samples with VOC refractive indices.
def generate_sample(vocs, definition, sample_index):
    sample_layers = []
    total_sample_thickness = DROPLET_DEPTH
    individual_voc_thickness = total_sample_thickness / definition
    adjusted_vocs = copy.deepcopy(vocs)

    for i in range(len(adjusted_vocs)):
        adjusted_vocs[i][2] = random.random() * 0.01

    adjusted_vocs[sample_index % len(vocs)][2] = 1
    total_voc_sum = max(sum([voc[2] for voc in adjusted_vocs]), 1)
    
    for voc in adjusted_vocs:
        num_voc = round((voc[2] / total_voc_sum) * definition)
        sample_layers.extend([[generate_refractive_index_from_csv(voc[3]), individual_voc_thickness, voc[0]]] * num_voc)
    
    random.shuffle(sample_layers)
    sample_layers.insert(0, [(1.0), None, None])
    sample_layers.append([(1.0), None, None])

    sample = [adjusted_vocs, sample_layers, []] # concentration, layers, reflection values

    return sample

# Generate samples and divide among processes.
start_gen = time.time()
if rank == 0:
    if VERBOSE > 0: print("Generating samples...")

local_samples = []

for i in range(round(NUM_SAMPLES // size)):
    local_samples.append(generate_sample(vocs, DEFINITION, i))
    if VERBOSE > 1: print(f"Generated sample {i} (rank {rank})...")

if VERBOSE > 0: print(f"Generated {len(local_samples)} samples in {(time.time() - start_gen):.3f}s...")

# Initialize lists for reflection and transmission.
local_R_s, local_T_s = [], []

# Process samples and wavelengths.
if VERBOSE > 0: print(f"Starting TMM processing (rank {rank})...")
for i in range(len(local_samples)):
    local_R_sample, local_T_sample = [], []
    sample = local_samples[i]
    sample_layers = sample[1]

    for w in wavelengths[:-1]:
        start = time.time()
        res = LightweightTransferMatrixMethod.solve_tmm(sample_layers, w, 0)
        if VERBOSE > 1: print(f"Calculated RT sample {i} with wavelength {w:.3f} nm in {time.time() - start:.3f}s (rank {rank})...")
        local_R_sample.append(res[0])
        local_T_sample.append(res[1])
    if VERBOSE > 1: print(f"Finished {((i + 1)/len(local_samples) * 100):0.2f}% of samples (rank {rank})...")

    local_R_s.append(local_R_sample)
    local_T_s.append(local_T_sample)
    sample[2] = list(1 - (np.array(local_R_sample) + np.array(local_T_sample)))

    local_samples[i] = {
        "wl": wavelengths[:-1].tolist(),
        "l": sample[0],
        "r": sample[2],
    }

    if SAVE_DATA:
        with open(f'data_mid_{rank}.json', 'w', encoding='utf-8') as f:
            json.dump(local_samples[:i + 1], f, ensure_ascii=False, indent=4)


# Gather results at the root process.
R_s = comm.gather(local_R_s, root=0)
T_s = comm.gather(local_T_s, root=0)
all_samples = comm.gather(local_samples, root=0)

# If root process, save and plot the results.
if rank == 0:
    all_samples = sum(all_samples, [])
    # Save results to JSON file.
    if SAVE_DATA:
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=4)
        if VERBOSE > 0: print("Stored reflectance and transmission values!")
    
    # Calculate and output total elapsed time.
    program_end = MPI.Wtime()
    elapsed_time = program_end - program_start
    if VERBOSE > 0: print(f"Total elapsed time: {elapsed_time:.3f} seconds")

    plt.plot(all_samples[0]['wl'], all_samples[0]['r'], label='Reflectance')
    plt.xlabel("Wavelength (in nm)")
    plt.legend()
    plt.savefig('data_1.png')