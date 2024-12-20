# Many thanks to https://towardsdatascience.com/data-science-for-raman-spectroscopy-a-practical-example-e81c56cf25f


# Loading the required packages:
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import random
import time

start = time.time()

NUM_COMPONENTS = 8

def Gauss(x, mu, sigma, A=1):    
    return A / (sigma * np.sqrt(2 * np.pi)) \
        * np.exp(-0.5 * ((x-mu) / sigma)**2)

def create_component(x_range, peaks):
    component = np.zeros_like(x_range)
    for mu, sigma, intensity in peaks:
        component += Gauss(x_range, mu, sigma, intensity)
    return component / np.max(component)

def cm1_to_nm(cm_1):
    return 1e7 / cm_1


random.seed(68)

x_range = np.linspace(780, 1400, 1000)

# Generate NUM_COMPONENTS components with random peaks
components = []
for i in range(NUM_COMPONENTS):
    num_peaks = random.randint(1, 3)
    peaks = [(random.uniform(780, 1200), random.uniform(1, 10), random.uniform(0.25, 0.75)) for _ in range(num_peaks)]
    component = create_component(x_range, peaks)
    components.append(component)

"""
components = [
    [   # CH4
        # https://www.researchgate.net/figure/Raman-spectra-of-CH4-The-Raman-spectrum-in-red-shows-one-prominent-signal-for-the-CH4_fig3_362645904
        cm1_to_nm(2917),
    ],
    [   # CO2
        # https://www.researchgate.net/publication/244732172_Micro-Raman_Thermometer_for_CO2_Fluids_Temperature_and_Density_Dependence_on_Raman_Spectra_of_CO2_Fluids
        cm1_to_nm(1388),
        cm1_to_nm(1285),
    ],
    [   # N2 Gas
        # https://www.researchgate.net/publication/364451331_Precision_evaluation_of_nitrogen_isotope_ratios_by_Raman_spectrometry
        cm1_to_nm(2300),
    ],
]


x_range = np.linspace(780, max(list([item for sublist in components for item in sublist ])) + 300, 8000) # IR-A Range to Far-IR (1m)

for i in range(len(components)):
    peaks = components[i]

    for j in range(len(peaks)):
        peaks[j] = (peaks[j], random.uniform(1, 10), random.uniform(0.35, 0.75))
    print(peaks)
    components[i] = create_component(x_range, peaks)
"""

NUM_COMPONENTS = len(components)
NUM_SAMPLES = 1000
concentrations = []
spectrums = []

poly = random.uniform(0.00001, 0.0001) * (0.2 * np.ones(len(x_range)) + 0.0001 * x_range + 0.000051 * (x_range - 680)**2) 

for i in range(NUM_SAMPLES):
    sample_concentration = []

    for j in range(NUM_COMPONENTS):
        sample_concentration.append(random.uniform(0.1, 0.5))

    sample_concentration = np.array(sample_concentration)
    sample_concentration = (1 / sample_concentration.sum()) * sample_concentration
    concentrations.append(sample_concentration.tolist())    

    spectrum = np.zeros_like(x_range)

    spectrum += sample_concentration[j] * components[j]

    spectrum += np.random.normal(0, 0.02, len(x_range))

    spectrum += poly

    spectrums.append(spectrum.tolist())
    print(f"Generated spectrum {i+1}/{NUM_SAMPLES}")

plt.plot(x_range, spectrums[13])
plt.savefig('spectrum.png')

print("Outputting data to data.json...")
with open('data.json', 'w') as f:
    json.dump({
        'range': x_range.tolist(),
        'spectrums': spectrums,
        'concentrations': concentrations
    }, f, indent=4)
print(f"Done in {time.time() - start:.2f} seconds!")