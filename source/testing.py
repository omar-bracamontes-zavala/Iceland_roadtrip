import os
import requests
import urllib.parse
from datetime import datetime
from typing import List, Tuple

def _parse_cities_to_str(cities: List[Tuple[float, float]]) -> str:
    # Convert coordinates to the required string format
    cities_str = '|'.join([f"{lat},{lng}" for lat, lng in cities])
    return urllib.parse.quote(cities_str)

def _generate_url(origins: List[Tuple[float, float]], destination: List[Tuple[float, float]],
                  travel_mode: str = 'driving', departure_time: str = 'now') -> str:
    # Your Google Maps API key
    api_key = os.environ.get('GOOGLE_API_KEY')
    base_url = 'https://maps.googleapis.com/maps/api/distancematrix/json?'
    
    origins_parsed = _parse_cities_to_str(origins)
    destinations_parsed = _parse_cities_to_str(destination)
    
    return f'{base_url}origins={origins_parsed}&destinations={destinations_parsed}&mode={travel_mode}&departure_time={departure_time}&key={api_key}'

def get_distance_matrix(origins: List[Tuple[float, float]], destination: List[Tuple[float, float]],
                        travel_mode: str = 'driving', departure_time: str = 'now'):
    url = _generate_url(origins, destination, travel_mode, departure_time)
    response = requests.get(url)
    distance_matrix = response.json()
    
    return distance_matrix
    


# # Define the locations with coordinates (latitude, longitude)
origins = [(40.712776, -74.005974), (34.052235, -118.243683)]  # New York, NY and Los Angeles, CA
destinations = [(37.774929, -122.419418), (41.878113, -87.629799)]  # San Francisco, CA and Chicago, IL

# # Convert coordinates to the required string format
# origins_str = '|'.join([f"{lat},{lng}" for lat, lng in origins])
# destinations_str = '|'.join([f"{lat},{lng}" for lat, lng in destinations])

# # Construct the API request URL
# url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={urllib.parse.quote(origins_str)}&destinations={urllib.parse.quote(destinations_str)}&mode=driving&departure_time=now&key={api_key}"

# Make the request
# response = requests.get(url)
# distance_matrix = response.json()

# Extract and print the distance and duration information
distance_matrix = get_distance_matrix(origins, destinations)


for origin_index, origin in enumerate(distance_matrix['origin_addresses']):
    for destination_index, destination in enumerate(distance_matrix['destination_addresses']):
        element = distance_matrix['rows'][origin_index]['elements'][destination_index]
        if element['status'] == 'OK':
            distance = element['distance']['text']
            duration = element['duration']['text']
            print(f"From {origin} to {destination}: {distance}, {duration}")
        else:
            print(f"From {origin} to {destination}: Not available")
