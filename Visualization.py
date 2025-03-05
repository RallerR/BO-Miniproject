import numpy as np
import matplotlib.pyplot as plt
import json

# 🔹 Set your JSON file name here
json_filename = "mount_fuji_elevation_data.json"  # Change this to switch between datasets

# 🔹 Choose whether to standardize the data
standardize = False  # Set to False to use raw elevation values

# 🔹 Choose standardization method if enabled
use_min_max = True  # Set to False to use Z-score standardization

# Load elevation data from JSON file
with open(json_filename, "r") as f:
    elevation_grid = json.load(f)

# Convert to numpy array for processing
elevation_array = np.array(elevation_grid)

# Apply standardization if enabled
if standardize:
    if use_min_max:
        # Min-Max Normalization (0 to 1 scaling)
        elevation_array = (elevation_array - np.min(elevation_array)) / (np.max(elevation_array) - np.min(elevation_array))
    else:
        # Z-score Standardization
        elevation_array = (elevation_array - np.mean(elevation_array)) / np.std(elevation_array)

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(elevation_array, cmap="terrain", origin="lower", interpolation='nearest')
plt.colorbar(label="Elevation" if not standardize else "Standardized Elevation")
plt.title(f"Elevation Heatmap ({'Standardized' if standardize else 'Raw'}) - {json_filename}")
plt.xlabel("Longitude Index")
plt.ylabel("Latitude Index")
plt.show()

# Find the highest elevation point
max_index = np.unravel_index(np.argmax(elevation_array), elevation_array.shape)
print(f"Highest elevation at index: {max_index}, Value: {elevation_array[max_index]}")
