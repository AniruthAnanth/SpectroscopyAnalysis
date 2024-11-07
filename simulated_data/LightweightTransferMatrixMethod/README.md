# A Lightweight Transfer Matrix Method in Python
A small Python library for calculating reflectance and transmittance data for 1 dimensional multilayer media.

## Example usage
```py
import numpy as np
import tmm

# Define refractive index functions for each layer
# These could represent, for instance, a dielectric layer and a metal layer
def n_layer1(wavelength):
    # Refractive index for layer 1 (e.g., 1.5 for a typical dielectric)
    return 1.5

def n_layer2(wavelength):
    # Refractive index for layer 2 (e.g., 2.5 for another material)
    return 2.5

# Define the stack of layers as tuples: (refractive index function, thickness in meters)
# For example, two layers of different materials, each with a given thickness
layers = [
    (lambda _: 1., 0),            # Starting medium (assumed air, refractive index of 1)
    (n_layer1, 500e-9),       # Layer 1: 500 nm thickness
    (n_layer2, 300e-9),       # Layer 2: 300 nm thickness
    (lambda _: 1., 0)             # Ending medium (assumed air, refractive index of 1)
]

# Define the wavelength and angle of incidence
wavelength = 600e-9  # Wavelength of 600 nm
theta = 0            # Normal incidence (0 radians)

# Calculate reflection (R) and transmission (T)
R, T = tmm.solve_tmm(layers, wavelength, theta)

# Print the results
print(f"Reflection (R): {R}")
print(f"Transmission (T): {T}")
```

## Future goals
- Integrate with mpi4py
- More dimensions
- Diagonally anisotropic & fully anisotropic media
- Scattering matrix version
