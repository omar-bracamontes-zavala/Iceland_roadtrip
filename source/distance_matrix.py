# Type Hints
from typing import List, Tuple, Dict, Any, Generator
# Main
import os, requests, urllib.parse, json
from helpers import read_cities

#
# Extras
#
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
      
#
# Helpers
#
def _parse_cities_to_str(cities: List[Tuple[float, float]]) -> str:
    """
    Convert a list of coordinate tuples to a URL-encoded string.

    Args:
        cities (List[Tuple[float, float]]): A list of tuples containing latitude and longitude coordinates.

    Returns:
        str: A URL-encoded string representing the list of coordinates.
    """
    cities_str = '|'.join([f"{lat},{lng}" for lat, lng in cities])
    return urllib.parse.quote(cities_str)

def _generate_url(origins: List[Tuple[float, float]], destinations: List[Tuple[float, float]], **kwargs) -> str:
    """
    Generate a URL for the Google Maps Distance Matrix API.

    Args:
        origins (List[Tuple[float, float]]): A list of tuples containing latitude and longitude coordinates for the origins.
        destinations (List[Tuple[float, float]]): A list of tuples containing latitude and longitude coordinates for the destinations.
        **kwargs: Additional optional parameters to include in the URL (e.g., mode, departure_time, traffic_model).
                  https://developers.google.com/maps/documentation/distance-matrix/distance-matrix

    Returns:
        str: A URL string for querying the Google Maps Distance Matrix API.
    """
    api_key = os.environ.get('GOOGLE_API_KEY')
    base_url = 'https://maps.googleapis.com/maps/api/distancematrix/json?'
    
    origins_parsed = _parse_cities_to_str(origins)
    destinations_parsed = _parse_cities_to_str(destinations)
    
    # Start building the URL with required parameters
    url = f'{base_url}origins={origins_parsed}&destinations={destinations_parsed}&key={api_key}'
    
    # Append any additional optional parameters from kwargs
    for key, value in kwargs.items():
        url += f'&{key}={urllib.parse.quote(str(value))}'
    
    return url

def _chunked_iterable(iterable: List, size: int) -> Generator:
    """
    Yield successive chunks of a given size from the iterable.

    Args:
        iterable (List): The list to be chunked.
        size (int): The size of each chunk.

    Yields:
        Generator: A generator that yields chunks of the given size.
    """
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]
    
#
# Main (maximum of 100 elements per request AND maximum 25 elements per origins/destinations)
#  
def request_distance_matrix(origins: List[Tuple[float, float]], destinations: List[Tuple[float, float]], **kwargs) -> Dict[str, Any]:
    """
    Get the distance matrix from the Google Maps Distance Matrix API.

    Args:
        origins (List[Tuple[float, float]]): A list of tuples containing latitude and longitude coordinates for the origins.
        destinations (List[Tuple[float, float]]): A list of tuples containing latitude and longitude coordinates for the destinations.
        **kwargs: Additional optional parameters to include in the URL (e.g., mode, departure_time, units, traffic_model).
                  https://developers.google.com/maps/documentation/distance-matrix/distance-matrix

    Returns:
        dict: The distance matrix data from the API response.
    
    Example Output:
        See: https://developers.google.com/maps/documentation/distance-matrix/distance-matrix#distance-matrix-advanced
    """
    url = _generate_url(origins, destinations, **kwargs)
    response = requests.get(url)
    raw_distance_matrix = response.json()
    
    # if save_file:
    #     _save_file_as_json(raw_distance_matrix, 'raw_distance_matrix.json')
    return raw_distance_matrix
  
def parse_distance_matrix(distance_matrix_response: Dict[str, Any]) -> List[List[Tuple[int, int, int]]]:
    """
    Parse the Google API distance matrix response.

    Args:
        distance_matrix_response (Dict[str, Any]): The response output from the Google API.

    Returns:
        List[List[Tuple[int, int, int]]]: Distance matrix where each entry means 
        (distance in meters, duration in seconds, duration in traffic in seconds [optional]).
        The rows are the origin and the columns are the destination.
    """
    distance_matrix = [
        [
            (
                element['distance']['value'] if element['status'] == 'OK' else None,
                element['duration']['value'] if element['status'] == 'OK' else None,
                element.get('duration_in_traffic', {}).get('value') if element['status'] == 'OK' else None
            )
            for element in origin['elements']
        ]
        for origin in distance_matrix_response['rows']
    ]
    # if save_file:
    #     _save_file_as_json(distance_matrix, 'distance_matrix.json')
        
    return distance_matrix

def generate_distance_matrix() -> List[List[Tuple[int, int, int]]]:
    """
    Generate a complete distance matrix for a list of cities.

    The function divides the list of cities into manageable chunks and requests
    distance matrices for these chunks. It then assembles these smaller matrices
    into a complete distance matrix.

    Returns:
        List[List[Tuple[int, int, int]]]: The complete distance matrix. ( distance [m], duratino [s], duration_in_traffic [s] )
    """
    cities = read_cities()
    max_total_cities_per_request = 100  # Maximum cities per request as per documentation
    max_destinations_per_request = 25   # Experimentally determined limit
    max_origins_per_request = max_total_cities_per_request // max_destinations_per_request  # 4

    complete_distance_matrix = []

    origin_chunks = list(_chunked_iterable(cities, max_origins_per_request))
    destination_chunks = list(_chunked_iterable(cities, max_destinations_per_request))

    for origins in origin_chunks:
        complete_distance_row = []

        for destinations in destination_chunks:
            distance_matrix_response = request_distance_matrix(
                origins=origins,
                destinations=destinations,
                # Defaults but useful to notice
                mode='driving',
                departure_time='now',
                units='metric',
                traffic_model='best_guess',
            )
            distance_matrix = parse_distance_matrix(distance_matrix_response)

            if not complete_distance_row:
                complete_distance_row = distance_matrix
            else:
                for i, row in enumerate(distance_matrix):
                    complete_distance_row[i].extend(row)

        complete_distance_matrix.extend(complete_distance_row)

    return complete_distance_matrix

if __name__=='__main__':
    distance_matrix = generate_distance_matrix()
    _save_file_as_json(distance_matrix, 'distance_matrix.json')
    print('Done!')
