import numpy as np
import math

def generate_centered_grid(center_lat, center_lng, grid_size_m, step_size_m):
    """
    Generates a grid of latitude/longitude points centered on a given coordinate.

    Parameters:
    - center_lat (float): Latitude of the center point.
    - center_lng (float): Longitude of the center point.
    - grid_size_m (float): Total width/height of the grid in meters.
    - step_size_m (float): Distance between points in meters.

    Returns:
    - List of (latitude, longitude) tuples forming a grid.
    """
    # Convert meters to degrees
    lat_step = step_size_m / 111000  # 1 degree latitude ≈ 111 km
    lng_step = step_size_m / (111000 * math.cos(math.radians(center_lat)))  # Adjust for longitude

    # Number of steps in each direction
    num_steps = int(grid_size_m / step_size_m) // 2  # Half on each side of the center

    # Generate grid points
    grid_points = [
        (center_lat + i * lat_step, center_lng + j * lng_step)
        for i in range(-num_steps, num_steps + 1)
        for j in range(-num_steps, num_steps + 1)
    ]

    return grid_points

# Example usage: Grid around Mount Everest (27.988767702702702, 86.92568360443653), 4km x 4km grid, 100m step
everest_grid = generate_centered_grid(27.988767702702702, 86.92568360443653, grid_size_m=4000, step_size_m=100)


