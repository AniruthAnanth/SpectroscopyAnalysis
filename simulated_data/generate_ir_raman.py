# Many thanks to https://towardsdatascience.com/data-science-for-raman-spectroscopy-a-practical-example-e81c56cf25f


# Loading the required packages:
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import time

random.seed(42)

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

ir_range = np.linspace(780, 1400, 1024) # IR-A Range

# Generate 20 components with random peaks
components = []
for i in range(20):
    num_peaks = random.randint(1, 5)
    peaks = [(random.uniform(780, 1400), random.uniform(0.5, 5), random.uniform(0.05, 1)) for _ in range(num_peaks)]
    component = create_component(ir_range, peaks)
    components.append(component)

def generate_ir_sample(component_data):
    mixture = component_data[0][0] * component_data[0][1]
    for i in range(1, len(component_data)):
        mixture += component_data[i][0] * component_data[i][1]

    mixture += np.random.normal(0, 0.01, len(ir_range))

    return mixture

def generate_from_concentrations(components, concentrations):
    all_component_data = []

    for i in range(0, len(concentrations)):
        start = time.time()
        concentration_set = concentrations[i]
        concentration = np.array(concentrations)
        concentration = (1 / np.sum(concentration) * concentration).tolist()
        component_data = []
        for component, concentration in zip(components, concentration_set):
            component_data.append((component, concentration))
        all_component_data.append(component_data)
        print(f"Generated concentrations {i + 1} in {time.time()-start:.4f} seconds ({(i + 1)/len(concentrations)*100:.2f}%)")
    
    return all_component_data

NUM_SAMPLES = 20000

concentrations = [
    [random.uniform(0.01, 0.5) for j in range(20)] for i in range(NUM_SAMPLES)
]

all_component_data = generate_from_concentrations(components, concentrations)

mix_spectrums = []

for i in range(len(all_component_data)):
    component = all_component_data[i]
    start = time.time()
    mix_spectrums.append(generate_ir_sample(component).tolist())
    print(f"Finalized sample {i} in {time.time() - start:.7f} second ({(i + 1)/len(all_component_data)*100:.2f}%)")

print("Outputting data to data.json...")
json.dump({
    'range': ir_range.tolist(),
    'spectrums': mix_spectrums,
    'concentrations': concentrations
}, open('data.json', 'w'), indent=4)
print("Done!")