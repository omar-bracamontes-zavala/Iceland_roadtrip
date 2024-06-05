'''
    Note: Each city has an unique location (lat, lon)
'''
# Type Hints
from typing import List, Tuple
# Processing
import pandas as pd
# Plot
import matplotlib.pyplot as plt
import folium
# Main
import numpy as np
from geopy.distance import geodesic
import random


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

def plot_cities(cities: List[Tuple[float, float]], assignments: List[int], centroids: List[int], map_or_scatter: str='scatter'):
    if map_or_scatter == 'scatter':
        # Unpacking the list of tuples into x and y coordinates
        x, y = zip(*cities)
        
        # Find the number of unique clusters
        num_clusters = len(set(assignments))
        
        # Generate a colormap with enough colors for the clusters
        cmap = plt.get_cmap('tab20', num_clusters)
        
        # Determine sizes for cities
        sizes = [100 if i in centroids else 30 for i in range(len(cities))]
        
        # Creating the scatter plot with cluster-based colors
        scatter = plt.scatter(x, y, c=assignments, cmap=cmap, s=sizes, edgecolor='k', alpha=0.6)

        # Plotting the centroids with full opacity and same color as their cluster
        for i in centroids:
            plt.scatter(x[i], y[i], c=[assignments[i]], cmap=cmap, alpha=1, s=150, edgecolor='k', marker='x')

        # Adding title and labels
        plt.title('Iceland')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        
        # Showing the plot
        plt.colorbar(scatter, label='Cluster', ticks=range(num_clusters), boundaries=np.arange(num_clusters+1)-0.5)
        plt.show()
        
    elif map_or_scatter=='map':
        # Calculate the mean of the latitudes and longitudes for the initial map center
        mean_lat = sum([point[0] for point in cities]) / len(cities)
        mean_lon = sum([point[1] for point in cities]) / len(cities)

        # Create a map centered around the average city
        map = folium.Map(city=[mean_lat, mean_lon], zoom_start=6)

        # Add markers to the map
        for lat, lon in cities:
            folium.Marker([lat, lon]).add_to(map)

        # Save the map as an HTML file
        map.save('map.html')

        print("Map has been saved to 'map.html'. Open this file in your web browser to view the map.")
   

def calculate_distance(city_1:Tuple[float, float], city_2:Tuple[float, float]):
    '''
        city_1: (lat, lon)
    '''
    return geodesic(city_1, city_2).km

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

#
# K-Means: K clusters as the trip duration days
#

# Centroids initialization (random)
def initialize_centroids(cities: List[Tuple[float, float]], k: int) -> List[int]:
    """
    Selects k random elements from a list of cities.

    Args:
    cities (List[Tuple[float, float]]): A list of tuples where each tuple represents a geographic city (latitude, longitude).
    k (int): The number of random cities to select.

    Returns:
    List[int]: A list of k randomly selected index cities.
    """
    if k > len(cities):
        raise ValueError("k cannot be greater than the number of cities in the list")
    return random.sample(range(0, len(cities)), k) 

# Assigment Step
def assign_datum_to_cluster(cities: List[Tuple[float, float]], centroids: List[int], distance_matrix: List[List[float]]) -> List[int]:
    """
    Assign each city to the nearest centroid based on a precomputed distance matrix.

    Args:
    cities (List[Tuple[float, float]]): A list of tuples, where each tuple contains coordinates of a city.
    centroids (List[int]): A list of integers, where each int is the centroid city index.
    distance_matrix (List[List[float]]): A matrix of distances where element [i][j] represents the distance from city i to centroid j.

    Returns:
    List[int]: A list where the index represents the index of a city in the input list, and the value at that index
    represents the index of the closest centroid in the centroids list.
    """
    assignments = []

    for city_index, _ in enumerate(cities):
        if city_index in centroids:
            assignments.append(city_index)
        else:
            distances = [distance_matrix[city_index][centroid] for centroid in centroids]
            min_distance_index = centroids[distances.index(min(distances))]
            assignments.append(min_distance_index)
    
    return assignments

# Update Step
# Convergence Check
#

if __name__=='__main__':
    k_clusters = 4 # days
    cities = read_cities()
    distance_matrix = create_distance_matrix(cities)
    # print(distance_matrix)
    centroids = initialize_centroids(cities, k_clusters)
    # print(centroids)
    assignments = assign_datum_to_cluster(cities, centroids, distance_matrix)
    # print(assignments)
    plot_cities(cities, assignments, centroids)