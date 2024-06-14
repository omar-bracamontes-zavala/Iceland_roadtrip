# Type Hints
from typing import List, Tuple
# Processing
import pandas as pd
# Main
from geopy.distance import geodesic

#
# Helpers
#
def read_cities(filepath: str = 'datasets/cities.csv') -> List[Tuple[float, float]]:
    """
    Loads geographic city data from a CSV file and returns a list of tuples.
    Each tuple contains latitude and longitude as floats.

    Args:
    filepath (str): The path to the CSV file.

    Returns:
    List[Tuple[float, float]]: A list of tuples with latitude and longitude.
    """
    # Load data using pandas
    df = pd.read_csv(filepath, usecols=[1, 2])

    # Convert DataFrame to list of tuples
    cities = list(df.itertuples(index=False, name=None))

    return cities

def calculate_distance(city_1:Tuple[float, float], city_2:Tuple[float, float]):
    '''
        city_1: (lat, lon)
    '''
    return geodesic(city_1, city_2).km
