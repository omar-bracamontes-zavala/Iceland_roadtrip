# Type Hints
from typing import List, Tuple
# Helpers
from helpers import calculate_distance


def create_distance_matrix(cities: List[Tuple[float, float]]) -> List[List[float]]:
    """
    Create a distance matrix where each row corresponds to a city and each column to a centroid.

    Args:
    cities (List[Tuple[float, float]]): A list of city coordinates.

    Returns:
    List[List[float]]: A matrix of distances where element [i][j] represents the distance from city i to coty j.
    """
    matrix = [ [calculate_distance(city_i, city_j) for city_j in cities] for city_i in cities]

    return matrix