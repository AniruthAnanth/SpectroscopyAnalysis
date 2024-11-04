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

NUM_WAVELENGTHS = 100
DEFINITION = 100000
DROPLET_DEPTH = 2000000

random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Capture start time
program_start = MPI.Wtime()

def save(material, wavelengths, filename):
    out_str = "wavelength,n,kappa"

    for wavelength in wavelengths:
        n, kappa = get_n_kappa(material, wavelength)
        out_str += f"\n{wavelength},{n},{kappa}"

    f_out = open(out_folder+filename, "w")
    f_out.write(out_str)
    f_out.close()


def get_n_kappa(material, wavelength):
    n = 0
    kappa = 0

    n = material.get_refractive_index(wavelength)
    try:
        kappa = material.get_extinction_coefficient(wavelength)
    except:
        kappa = 0
    return max(n, 0), max(kappa, 0)


def find_voc_in_db(voc_name):
    organic_path = database_path + "data-nk/organic/"
    main_path = database_path + "data-nk/main/"

    organic_filenames = os.listdir(organic_path)
    main_filenames = os.listdir(main_path)
    voc_filename = None
    voc_name.lower()

    for filename in organic_filenames:
        if voc_name in filename.split(' - '):
            voc_filename = organic_path + filename
            break

    if not voc_filename:
        for filename in main_filenames:
            if voc_name in filename.split(' - '):
                voc_filename = main_path + filename
                break

    if not voc_filename:
        raise Exception(
            f"{voc_name} does not exist. All possible organic compounds listed below: \n{organic_filenames}")

    voc_page_names = os.listdir(voc_filename)
    voc_filename, voc_shelf = voc_filename.split(
        '/')[-1], voc_filename.split('/')[-2]

    for i in range(len(voc_page_names)):
        voc_page_names[i] = voc_page_names[i].replace('.yml', '')

    return voc_shelf, voc_filename, voc_page_names

def generate_refractive_index_from_csv(csv):
    df = pd.read_csv(csv)
    lbdas = df['wavelength'][1:].to_numpy().astype(np.float64)[0:]
    n_s = df['n'][1:].to_numpy().astype(np.float64)[0:]
    k_s = df['kappa'][1:].to_numpy().astype(np.float64)[0:]

    def raw_data_points(lbda):
        i = df[df['wavelength'] == lbda].index[0]
        return n_s[i] - 1j * k_s[i]

    raw_data_points.__name__ = csv + " refractive index"

    return raw_data_points

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

# Only root loads VOC refractive indices
if rank == 0:
    print("Loading VOC refractive indices...")
    for i in range(len(vocs)):
        voc_shelf, voc_book_name, voc_page_names = find_voc_in_db(vocs[i][0])
        voc_material = RefractiveIndexMaterial(shelf=voc_shelf, book=voc_book_name, page=voc_page_names[vocs[i][1]])
        save(voc_material, wavelengths, voc_book_name + ".csv")
        vocs[i][3] = out_folder + voc_book_name + ".csv"
        print(f"Loaded {voc_book_name}...")
    print("Loaded all VOC's!")

# Broadcast VOC data to all processes
vocs = comm.bcast(vocs, root=0)

def generate_sample(vocs, definition):
    sample = []
    total_sample_thickness = DROPLET_DEPTH
    individual_voc_thickness = total_sample_thickness / definition
    total_voc_sum = max(sum([voc[2] for voc in vocs]), 1)
    for voc in vocs:
        num_voc = round((voc[2] / total_voc_sum) * definition)
        s = generate_refractive_index_from_csv(voc[3])
        sample.extend([[generate_refractive_index_from_csv(voc[3]), individual_voc_thickness, voc[0]]] * num_voc)
    random.shuffle(sample)
    sample.insert(0, [(1.0), None])
    sample.append([(1.0), None])
    return sample

test_sample = generate_sample(vocs, DEFINITION)

# Divide wavelengths among processes
local_wavelengths = np.array_split(wavelengths[:-1], size)[rank]
local_R_s, local_T_s = [], []

start = time.time()
for w in local_wavelengths:
    start_in = time.time()
    res = LightweightTransferMatrixMethod.solve_tmm(test_sample, w, 0)
    local_R_s.append(res[0])
    local_T_s.append(res[1])
    print(f"Rank {rank} calculated for {w:0.4f} nm in {(time.time() - start_in):0.3f}s")

# Gather results at root process
R_s = comm.gather(local_R_s, root=0)
T_s = comm.gather(local_T_s, root=0)

if rank == 0:
    # Flatten lists of lists
    R_s = [item for sublist in R_s for item in sublist]
    T_s = [item for sublist in T_s for item in sublist]
    
    # Plot results
    plt.plot(wavelengths[:-1], R_s, label="Reflection")
    plt.xlabel("Wavelength in nm")
    plt.legend()
    plt.show()
    
    # Save results to JSON
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump([wavelengths[:-1].tolist(), R_s], f, ensure_ascii=False, indent=4)

    program_end = MPI.Wtime()
    elapsed_time = program_end - program_start
    print(f"Total elapsed time: {elapsed_time:.3f} seconds")