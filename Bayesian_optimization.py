import numpy as np
import matplotlib.pyplot as plt
import json
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.acquisition import gaussian_ei
from Dataset_creation import generate_centered_grid

np.random.seed(2)

# ---------------------------
# Define dataset parameters in a dictionary
# ---------------------------
datasets = {
    "DTU": {
         "filename": "dtu_elevation_data.json",
         "center_lat": 55.785734,
         "center_lng": 12.521527,
         "grid_size_m": 1000,
         "step_size_m": 25,
    },
    "Mount Everest": {
         "filename": "mount_everest_elevation_data.json",
         "center_lat": 27.988767702702702,
         "center_lng": 86.92568360443653,
         "grid_size_m": 4000,
         "step_size_m": 100,
    },
    "Mount Fuji": {
         "filename": "mount_fuji_elevation_data.json",
         "center_lat": 35.362600,
         "center_lng": 138.730578,
         "grid_size_m": 1000,
         "step_size_m": 25,
    },
    "Grand Canyon": {
         "filename": "grand_canyon_elevation_data.json",
         "center_lat": 36.099758,
         "center_lng": -112.112486,
         "grid_size_m": 5000,
         "step_size_m": 125,
    },
    "Hawaii": {
         "filename": "hawaii_elevation_data.json",
         "center_lat": 20.825666,
         "center_lng": -156.926865,
         "grid_size_m": 500000,
         "step_size_m": 12500,
    }
}

# Choose a dataset by its key:
dataset_name = "Grand Canyon"  # e.g. "DTU", "Mount Fuji", "Grand Canyon", "Hawaii"
params = datasets[dataset_name]

# ---------------------------
# 1) SETUP: Load data, generate lat/lng grid
# ---------------------------
filename = params["filename"]
with open(filename, "r") as f:
    elevation_grid = json.load(f)

elevation_array = np.array(elevation_grid)
grid_size = elevation_array.shape[0]

center_lat = params["center_lat"]
center_lng = params["center_lng"]
grid_size_m = params["grid_size_m"]
step_size_m = params["step_size_m"]

grid_points = generate_centered_grid(center_lat, center_lng, grid_size_m, step_size_m)
latitudes  = np.array([p[0] for p in grid_points]).reshape(grid_size, grid_size)
longitudes = np.array([p[1] for p in grid_points]).reshape(grid_size, grid_size)

# ---------------------------
# 0) VISUALIZE RAW DATA (HEATMAP) in REAL ELEVATION
#     (We keep the colorbar for the raw data)
# ---------------------------
plt.figure(figsize=(8, 6))
raw_contour = plt.contourf(longitudes, latitudes, elevation_array, levels=50, cmap="viridis")
plt.colorbar(raw_contour, label="Elevation (m)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Elevation Heatmap - {dataset_name}")
#plt.show()
plt.savefig(f"{dataset_name}_heatmap.png", dpi=300)
plt.close()

# ---------------------------
# 0.5) NORMALIZE THE DATA
# ---------------------------
elev_min = elevation_array.min()
elev_max = elevation_array.max()
elevation_array_norm = (elevation_array - elev_min) / (elev_max - elev_min)

# ---------------------------
# 1.5) Define search space, objective uses normalized data
# ---------------------------
search_space = [
    Real(longitudes.min(), longitudes.max(), name="lng"),
    Real(latitudes.min(), latitudes.max(), name="lat")
]

@use_named_args(search_space)
def objective(lng, lat):
    x_idx = np.abs(longitudes[0, :] - lng).argmin()
    y_idx = np.abs(latitudes[:, 0] - lat).argmin()
    # Return negative of normalized elevation => we "maximize" normalized elevation
    return -elevation_array_norm[y_idx, x_idx]

# Initial guesses (corners)
Xinit = [
    [longitudes[0, 0],   latitudes[0, 0]],
    [longitudes[0, -1],  latitudes[0, -1]],
    [longitudes[-1, 0],  latitudes[-1, 0]],
    [longitudes[-1, -1], latitudes[-1, -1]]
]

# ---------------------------
# 2) BAYESIAN OPTIMIZATION on NORMALIZED data
# ---------------------------
res = gp_minimize(objective, search_space, n_calls=20, x0=Xinit, acq_func="gp_hedge")

# Convert best found normalized value -> real elevation
best_lng, best_lat = res.x
best_norm_elev = -res.fun            # best is negative of objective
best_real_elev = best_norm_elev*(elev_max - elev_min) + elev_min

print(f"\nBayesOpt best location: lat={best_lat:.6f}, lng={best_lng:.6f}")
print(f"BayesOpt best normalized elev: {best_norm_elev:.3f}")
print(f"BayesOpt best real elevation: {best_real_elev:.2f} m")

# ---------------------------
# 3) PLOT ELEVATION + BO PATH (Contour in Normalized scale)
#     --> We do NOT add a colorbar
#     --> We DO show the path, but if you also want to hide that,
#        comment out the lines that plot the path or the text.
# ---------------------------
plt.figure(figsize=(8, 6))
contour = plt.contourf(longitudes, latitudes, elevation_array_norm, levels=50, cmap="viridis")
# No colorbar:
# plt.colorbar(label="Normalized Elevation")

x_iters = np.array(res.x_iters)  # shape (N, 2)

# Plot the BO path with no label
plt.plot(x_iters[:, 0], x_iters[:, 1], 'ro:')

# Add iteration numbers but no label
for i, (lng, lat) in enumerate(x_iters):
    plt.text(lng, lat, f"{i+1}", color="black", fontsize=8)

# Omit plt.legend() entirely, so no box is drawn.

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Bayesian Optimization - {dataset_name}")
#plt.show()
plt.savefig(f"{dataset_name}_BO.png", dpi=300)
plt.close()

# ---------------------------
# 4) SURROGATE MODEL & ACQUISITION FUNCTION PLOTS
#     --> Remove the BO points from these two plots
# ---------------------------
# Extract final data
X = np.array(res.x_iters)
y = np.array(res.func_vals)  # negative of normalized elevation

gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=5, random_state=42)
gp.fit(X, y)

n_points = 100
lng_vals = np.linspace(longitudes.min(), longitudes.max(), n_points)
lat_vals = np.linspace(latitudes.min(), latitudes.max(), n_points)
mesh_lng, mesh_lat = np.meshgrid(lng_vals, lat_vals)
X_eval = np.vstack([mesh_lng.ravel(), mesh_lat.ravel()]).T

# ---------------------------
# 4.1) Acquisition Function (no colorbar, no BO points)
# ---------------------------
y_opt = np.min(y)  # 'minimizing' negative of normalized
acq_vals = gaussian_ei(X_eval, gp, xi=0.01, y_opt=y_opt)
acq_vals_grid = acq_vals.reshape(mesh_lng.shape)

plt.figure(figsize=(8, 6))
plt.contourf(mesh_lng, mesh_lat, acq_vals_grid, levels=50, cmap="magma")
# No colorbar, no scatter points:
# plt.colorbar(label="Acquisition Value (EI)")
# plt.scatter(X[:, 0], X[:, 1], c="cyan", edgecolors="black", label="Sampled Points")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Acquisition Function after final iteration")
# plt.legend()  # also remove the legend if not needed
#plt.show()
plt.savefig(f"{dataset_name}_acquisition_function.png", dpi=300)
plt.close()

# ---------------------------
# 4.2) Surrogate Function (no colorbar, no BO points)
# ---------------------------
mean_pred, std_pred = gp.predict(X_eval, return_std=True)
pred_norm_elev = -mean_pred  # objective was negative of normalized
pred_norm_elev_grid = pred_norm_elev.reshape(mesh_lng.shape)

plt.figure(figsize=(8, 6))
plt.contourf(mesh_lng, mesh_lat, pred_norm_elev_grid, levels=50, cmap="viridis")
# No colorbar, no scatter points:
# plt.colorbar(label="Predicted Norm Elevation (Surrogate)")
# plt.scatter(X[:, 0], X[:, 1], c="white", edgecolors="black", label="Sampled Points")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Surrogate Function after final iteration")
# plt.legend()
#plt.show()
plt.savefig(f"{dataset_name}_surrogate_function.png", dpi=300)
plt.close()

# ---------------------------
# 5) RANDOM SAMPLING (20 SAMPLES) FOR COMPARISON
# ---------------------------
N_random = 20
random_lng = np.random.uniform(low=longitudes.min(), high=longitudes.max(), size=N_random)
random_lat = np.random.uniform(low=latitudes.min(), high=latitudes.max(), size=N_random)

random_obj_vals = []
for lng, lat in zip(random_lng, random_lat):
    val = objective([lng, lat])  # negative of normalized elev
    random_obj_vals.append(val)

random_obj_vals = np.array(random_obj_vals)
best_random_idx = np.argmin(random_obj_vals)
best_random_obj = random_obj_vals[best_random_idx]
best_random_lng = random_lng[best_random_idx]
best_random_lat = random_lat[best_random_idx]
best_random_norm = -best_random_obj
best_random_real = best_random_norm*(elev_max - elev_min) + elev_min

print("\n--- Random Sampling (20 samples) ---")
print(f"Best found location: lat={best_random_lat:.6f}, lng={best_random_lng:.6f}")
print(f"Best normalized elev: {best_random_norm:.3f}")
print(f"Best real-world elev: {best_random_real:.2f} m")

print("\n--- Comparison ---")
print(f"BO best real-world elev:      {best_real_elev:.2f} m")
print(f"Random best real-world elev:  {best_random_real:.2f} m")

# ---------------------------
# 6) FIND TRUE HIGHEST ELEVATION FOR COMPARISON
# ---------------------------
true_max_idx = np.unravel_index(np.argmax(elevation_array), elevation_array.shape)
true_max_elev = elevation_array[true_max_idx]  # True highest elevation
true_max_lat = latitudes[true_max_idx]
true_max_lng = longitudes[true_max_idx]

print("\n--- True Maximum Elevation ---")
print(f"Highest elevation in dataset: {true_max_elev:.2f} m")
print(f"Location: lat={true_max_lat:.6f}, lng={true_max_lng:.6f}")

# ---------------------------
# 7) FINAL COMPARISON
# ---------------------------
print("\n--- Final Comparison ---")
print(f"True highest elevation:        {true_max_elev:.2f} m")
print(f"BO best real-world elevation:  {best_real_elev:.2f} m")
print(f"Random best real-world elev:   {best_random_real:.2f} m")