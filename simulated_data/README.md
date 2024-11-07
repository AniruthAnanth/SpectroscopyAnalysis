# Simulated VOC Reflectance Analysis
This project calculates the reflection and transmission properties of volatile organic compounds (VOCs) by leveraging the Transfer Matrix Method (TMM) and parallel computing with MPI. The project simulates the optical properties of layered samples based on VOC refractive indices.

## Features
- **Refractive Index Data**: Loads refractive index and extinction coefficient data for various VOCs.
- **Parallel Computation**: Utilizes MPI to distribute wavelengths across multiple processes for efficient computation.
- **TMM Simulation**: Computes reflection and transmission values for specified wavelengths using the Transfer Matrix Method.
- **Output and Visualization**: Saves results to CSV and JSON files, with optional plots for visual analysis.

## Usage
Run the code with:
```
mpirun --use-hwthread-cpus python3 generate_reflectance_values.py
```