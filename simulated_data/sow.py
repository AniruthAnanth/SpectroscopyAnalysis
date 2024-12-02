import os
import json

# Directory containing the data_mid JSON files
data_dir = './'

# List to store all the data
all_data = []

# Iterate over all files in the directory
for filename in os.listdir(data_dir):
    if filename.startswith('data_mid') and filename.endswith('.json'):
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r') as file:
            print(file_path)
            data = json.load(file)
            all_data.extend(data)

# Write the concatenated data to sowed_data.json
print(len(all_data))
output_file = os.path.join(data_dir, 'sowed_data.json')
with open(output_file, 'w') as file:
    json.dump(all_data, file, indent=4)

print(f"Data from all data_mid files has been concatenated and stored in {output_file}")