# Type Hints
from typing import List, Tuple
# Processing
import pandas as pd
# Main
import numpy as np
from geopy.distance import geodesic
import random

#
# Helpers
#
def get_locations(filepath: str = 'datasets/locations.csv') -> List[Tuple[float, float]]:
    """
    Loads geographic location data from a CSV file and returns a list of tuples.
    Each tuple contains latitude and longitude as floats.

    Args:
    filepath (str): The path to the CSV file.

    Returns:
    List[Tuple[float, float]]: A list of tuples with latitude and longitude.
    """
    # Load data using pandas
    df = pd.read_csv(filepath, usecols=[1, 2])

    # Convert DataFrame to list of tuples
    locations = list(df.itertuples(index=False, name=None))

    return locations


#
# K-Means: K clusters as the trip duration days
#

# Centroids initialization (random)
def random_locations(locations: List[Tuple[float, float]], k: int) -> List[Tuple[float, float]]:
    """
    Selects k random elements from a list of locations.

    Args:
    locations (List[Tuple[float, float]]): A list of tuples where each tuple represents a geographic location (latitude, longitude).
    k (int): The number of random locations to select.

    Returns:
    List[Tuple[float, float]]: A list of k randomly selected locations.
    """
    if k > len(locations):
        raise ValueError("k cannot be greater than the number of locations in the list")
    return random.sample(locations, k)

# Assigment Step
# Update Step
# Convergence Check
#

if __name__=='__main__':
    k_clusters = 14 # days
    locations = get_locations()
    # print(locations)
    centroids = random_locations(locations, k_clusters)
    print(centroids)