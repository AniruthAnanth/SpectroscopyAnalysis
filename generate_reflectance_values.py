# Import necessary libraries for parallel processing, plotting, file handling, and mathematical operations.
from mpi4py import MPI
import matplotlib.pyplot as plt
import time
import LightweightTransferMatrixMethod
import os
import yaml
import scipy
import numpy as np
import pandas as pd
from ri import RefractiveIndexMaterial
import random
import json
import copy

# Define constants for the number of wavelengths, precision, and sample thickness.
NUM_WAVELENGTHS = 100#0
DEFINITION = 100000
DROPLET_DEPTH = 2000000
NUM_SAMPLES = 16  # Set the number of samples

# Seed the random number generator for reproducibility.
random.seed(42)

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

# Function to generate refractive index data from a CSV file.
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
database_path = "./refractiveindex.info-database/database/"
wavelengths = np.linspace(529, 585, NUM_WAVELENGTHS + 1)
vocs = [
    ["C2H4", 0, 0.00005, None],                 # Ethylene
    ["C2H4O2", 0, 0.00001, None],               # Acetic acid
    ["C2H6", 0, 0.00001, None],                 # Ethane
    ["C2H6O", 1, 0.00006, None],                # Ethanol
    ["C3H6O", 1, 0.00007, None],                # Acetone
    ["C3H8O", 1, 0.00002, None],                # Propanol
    ["C4H8O2", 1, 0.00003, None],               # Ethyl acetate
    ["C6H6", 1, 0.00002, None],                 # Benzene
    ["C7H8", 1, 0.00006, None],                 # Toluene
    ["C8H10", 0, 0.00003, None],                # Xylene
    ["CH4", 0, 0.00017, None],                  # Methane
    ["CH4O", 0, 0.00005, None],                 # Methanol
]

# If the current process is the root, load VOC refractive indices and save to files.
if rank == 0:
    print("Loading VOC refractive indices...")
    for i in range(len(vocs)):
        voc_shelf, voc_book_name, voc_page_names = find_voc_in_db(vocs[i][0])
        voc_material = RefractiveIndexMaterial(shelf=voc_shelf, book=voc_book_name, page=voc_page_names[vocs[i][1]])
        save(voc_material, wavelengths, voc_book_name + ".csv")
        vocs[i][3] = out_folder + voc_book_name + ".csv"
        print(f"Loaded {voc_book_name}...")
    print("Loaded all VOC's!")

# Broadcast the VOC data to all processes.
vocs = comm.bcast(vocs, root=0)

# Function to generate a batch of samples with VOC refractive indices.
def generate_sample(vocs, definition):
    sample_layers = []
    total_sample_thickness = DROPLET_DEPTH
    individual_voc_thickness = total_sample_thickness / definition
    adjusted_vocs = copy.deepcopy(vocs)

    for i in range(len(adjusted_vocs)):
        adjusted_vocs[i][2] += random.random() * 1e-4

    total_voc_sum = max(sum([voc[2] for voc in adjusted_vocs]), 1)
    
    for voc in adjusted_vocs:
        num_voc = round((voc[2] / total_voc_sum) * definition)
        sample_layers.extend([[generate_refractive_index_from_csv(voc[3]), individual_voc_thickness, voc[0]]] * num_voc)
    
    random.shuffle(sample_layers)
    sample_layers.insert(0, [(1.0), None])
    sample_layers.append([(1.0), None])

    sample = [adjusted_vocs, sample_layers, None] # concentration, layers, reflection values

    return sample

start_gen = time.time()
if rank == 0:
    print("Generating samples...")
# Generate samples and store them in a list.
all_samples = [generate_sample(vocs, DEFINITION) for _ in range(round(NUM_SAMPLES))]
if rank == 0:
    print(f"Generated {len(all_samples)} samples in {(time.time() - start_gen):.3f}s (rank {rank})...")

# Divide wavelengths among processes and initialize lists for reflection and transmission.
local_wavelengths = np.array_split(wavelengths[:-1], size)[rank]
local_R_s, local_T_s = [], []

print(f"Starting tmm processing (rank {rank})...")

# Loop over each sample and wavelength, compute reflection and transmission using TMM.
for i in range(len(all_samples)):
    local_R_sample, local_T_sample = [], []
    sample_layers = all_samples[i][1]

    for w in local_wavelengths:
        start = time.time()
        res = LightweightTransferMatrixMethod.solve_tmm(sample_layers, w, 0)
        print(f"Calculated RT with sample #{i} for {w:0.3f} nm in {(time.time() - start):0.3f}s (rank {rank})...")
        local_R_sample.append(res[0])
        local_T_sample.append(res[1])
    local_R_s.append(local_R_sample)
    local_T_s.append(local_T_sample)
    all_samples[i][2] = local_R_s

# Gather results at root process.
R_s = comm.gather(local_R_s, root=0)
T_s = comm.gather(local_T_s, root=0)

# If root process, save and plot the results.
if rank == 0:
    #R_s = [item for sublist in R_s for item in sublist]
    #T_s = [item for sublist in T_s for item in sublist]
    
    print("Processing results...")
    res = {}

    for i in range(len(all_samples)):
        res[i] = {
            "wl": wavelengths.tolist(),
            "l": [(i[2], i[1]) for i in all_samples[i][1][1:-1]],
            "r": all_samples[i][2],
        }


    # Save results to JSON file.
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    print("Stored reflectance and transmission values!")
    
    # Calculate and print total elapsed time.
    program_end = MPI.Wtime()
    elapsed_time = program_end - program_start
    print(f"Total elapsed time: {elapsed_time:.3f} seconds")