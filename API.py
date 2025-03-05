import requests
import time
import json
from Dataset_creation import generate_centered_grid

API_KEY = "AIzaSyBGdA07gn5mXni6Ym-hAdO69dDwsDpCrHA"  # Replace with your actual API Key
BASE_URL = "https://maps.googleapis.com/maps/api/elevation/json"
MAX_URL_LENGTH = 16000  # Slightly below 16,384 to be safe


def batch_api_call(coordinate_list, delay=1):
    """
    Calls the Google Elevation API in batches, ensuring the URL stays within length limits.

    Parameters:
        coordinate_list (list): List of (lat, lng) tuples.
        delay (float): Delay in seconds between API calls.

    Returns:
        list: Elevation data maintaining the input order.
    """
    elevations = []
    start_idx = 0

    while start_idx < len(coordinate_list):
        # Start with a conservative batch size and increase if possible
        batch_size = 512
        while batch_size > 1:
            batch = coordinate_list[start_idx:start_idx + batch_size]
            location_str = "|".join([f"{lat},{lng}" for lat, lng in batch])
            url = f"{BASE_URL}?locations={location_str}&key={API_KEY}"

            # If URL length is within limits, break the loop
            if len(url) < MAX_URL_LENGTH:
                break
            else:
                batch_size //= 2  # Reduce batch size if URL is too long

        print(f"Fetching {len(batch)} points (URL length: {len(url)})")

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data["status"] == "OK":
                elevations.extend([result["elevation"] for result in data["results"]])
                start_idx += len(batch)
            else:
                print(f"API error: {data['status']} - {data.get('error_message', '')}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

        time.sleep(delay)  # Avoid hitting API limits

    return elevations


def fetch_elevation_data(grid):
    """
    Fetches elevation data for a structured grid.

    Parameters:
        grid (list of lists): 2D list of (lat, lng) tuples.

    Returns:
        list of lists: 2D list of elevation values preserving grid structure.
    """
    flat_coords = [point for row in grid for point in row]

    elevations = batch_api_call(flat_coords)
    if elevations is None:
        print("Failed to fetch elevation data.")
        return None

    # Reshape elevation data to match original grid structure
    grid_size = len(grid[0])
    reshaped_elevations = [elevations[i * grid_size:(i + 1) * grid_size] for i in range(len(grid))]

    return reshaped_elevations

def process_location(location_name, center_lat, center_lng, grid_size_m=4000, step_size_m=100):
    """
    Fetches elevation data for a given location and saves it.

    Parameters:
        location_name (str): Name of the location.
        center_lat (float): Latitude of the center point.
        center_lng (float): Longitude of the center point.
        grid_size_m (int): Total grid size in meters.
        step_size_m (int): Step size in meters.

    Returns:
        None
    """
    print(f"Processing elevation data for: {location_name}")

    grid = generate_centered_grid(center_lat, center_lng, grid_size_m, step_size_m)

    grid_size = int(len(grid) ** 0.5)
    structured_grid = [grid[i * grid_size: (i + 1) * grid_size] for i in range(grid_size)]

    elevation_grid = fetch_elevation_data(structured_grid)

    if elevation_grid:
        filename = f"{location_name.lower().replace(' ', '_')}_elevation_data.json"
        with open(filename, "w") as f:
            json.dump(elevation_grid, f, indent=2)
        print(f"Elevation data saved successfully as {filename}")


# Example usage:
if __name__ == "__main__":
    # **Mount Everest (Highest Point)**
    #process_location("Mount Everest", 27.988767702702702, 86.92568360443653, 4000, 100)

    # **DTU**
    # process_location("DTU", 55.785734, 12.521527, 1000, 25)

    # **Grand Canyon**
    # process_location("Grand Canyon", 36.099758, -112.112486, 5000, 125)

    # **Hawaii**
    # process_location("Hawaii", 20.825666, -156.926865, 500000, 12500)

    # **Mount Fuji**
    # process_location("Mount Fuji", 35.362600, 138.730578, 1000, 25)

    print("Hello World")
