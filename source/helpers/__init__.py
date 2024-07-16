# Type Hints
from typing import List, Tuple
# Processing
import json
import pandas as pd
import numpy as np

#
# Helpers
#
def read_cities(filepath: str = 'source/datasets/cities.csv') -> List[Tuple[float, float]]:
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

    return np.array(cities)

def _save_file_as_json(distance_matrix: List[List[Tuple[int, int, int]]], filename: str) -> None:
    """
    Save the distance matrix to a JSON file.

    Args:
        distance_matrix (List[List[Tuple[int, int, int]]]): The distance matrix to save.
        filename (str): The name of the JSON file to save the matrix in.
    """
    filepath = f'source/datasets/{filename}'
    with open(filepath, 'w') as file:
        json.dump(distance_matrix, file)
  
def load_distance_matrix_from_json(filename: str) -> List[List[Tuple[int, int, int]]]:
    """
    Load the distance matrix from a JSON file.

    Args:
        filename (str): The name of the JSON file to load the matrix from.

    Returns:
        List[List[Tuple[int, int, int]]]: The loaded distance matrix.
    """
    filepath = f'source/datasets/{filename}'
    with open(filepath, 'r') as file:
        distance_matrix = json.load(file)
    return distance_matrix
 
