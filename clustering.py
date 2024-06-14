'''
    Note: Each city has an unique location (lat, lon)
'''
# Type Hints
from typing import List, Tuple
# Plot
import matplotlib.pyplot as plt
import folium
# Main
from helpers import calculate_distance, read_cities
import numpy as np
import random, json


#
# Plots
#
def plot_cities(cities: List[Tuple[float, float]], assignments: List[int], centroids: List[Tuple[float, float]], map_or_scatter: str='scatter', autoclose: bool=True):
    if map_or_scatter == 'scatter':
        # Unpacking the list of tuples into x and y coordinates
        x, y = zip(*cities)
        
        # Find the number of unique clusters
        num_clusters = len(set(assignments))
        
        # Generate a colormap with enough colors for the clusters
        cmap = plt.get_cmap('tab20', num_clusters)
        
        # Creating the scatter plot with cluster-based colors
        scatter = plt.scatter(x, y, c=assignments, cmap=cmap, s=30, edgecolor='k', alpha=0.6)

        # Plotting the centroids with full opacity and same color as their cluster
        for i,(x,y) in enumerate(centroids):
            plt.scatter(x, y, c=[i], cmap=cmap, alpha=1, s=150, marker='x')

        # Adding title and labels
        plt.title('Iceland')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        
        # Showing the plot
        plt.colorbar(scatter, label='Cluster', ticks=range(num_clusters), boundaries=np.arange(num_clusters+1)-0.5)
        if autoclose:
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()
        else:
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


#
# K-Means: K clusters as the trip duration days
#

# Centroids initialization (random)
def initialize_centroids(cities: List[Tuple[float, float]], k: int) -> List[Tuple[float, float]]:
    """
    Selects k random elements from a list of cities.

    Args:
    cities (List[Tuple[float, float]]): A list of tuples where each tuple represents a geographic city (latitude, longitude).
    k (int): The number of random cities to select.

    Returns:
    List[float, float]: A list of k randomly selected cities.
    """
    if k > len(cities):
        raise ValueError("k cannot be greater than the number of cities in the list")
    return random.sample(cities, k) 

# Assigment Step
def assign_datum_to_cluster(cities: List[Tuple[float, float]], centroids: List[Tuple[float, float]]) -> List[int]:
    """
    Assign each city to the nearest centroid by calculating the squared Euclidean distance to each centroid.
    A city is automatically assigned to itself if it is also a centroid.

    Args:
    cities (List[Tuple[float, float]]): A list of tuples, where each tuple contains coordinates of a city.
    centroids (List[Tuple[float, float]]): A list of tuples, where each tuple contains coordinates of a centroid.

    Returns:
    List[int]: A list where the index represents the index of a city in the cities input list, and the value at that index
    represents the index of the closest centroid in the centroids input list.
    """
    assignments = []
    
    for city in cities:
        current_min_distance = np.inf
        assigned_centroid = None
        
        for centroid_idx, centroid in enumerate(centroids):
            if (distance := calculate_distance(city, centroid)) < current_min_distance:
                current_min_distance = distance
                assigned_centroid = centroid_idx
        assignments.append(assigned_centroid)
    
    return assignments

# Update Step
def _gather_cities_by_cluster(cities: List[Tuple[float, float]], assignments: List[int]) -> List[Tuple[float, float]]:
    clusters = {}
    
    for city_index, cluster_index in enumerate(assignments):
        # Initialize
        if not clusters.get(cluster_index):
            clusters[cluster_index] = []
        
        clusters[cluster_index].append(cities[city_index])
    
    return clusters

def get_cluster_mean(cities: List[Tuple[float, float]], assignments: List[int]) -> List[Tuple[float, float]]:
    
    new_centroids = []
    clusters = _gather_cities_by_cluster(cities, assignments)
    
    for clustered_cities in clusters.values():
        lon, lat = zip(*clustered_cities)
        new_centroids.append( ( np.mean(lon), np.mean(lat) ))
    
    return new_centroids

# Termination Criteria
def termination_criteria(old_centroids: List[Tuple[float, float]], new_centroids: List[Tuple[float, float]], tolerance=1e-6):
    differences = [calculate_distance(old_centroid, new_centroids[i]) for i, old_centroid in enumerate(old_centroids)]
    # print(differences)
    max_difference = max(differences)
    return max_difference < tolerance
        
# Clustering
def k_means(k_clusters: int, cities: List[Tuple[float, float]], iterations: int):
    
    centroids = initialize_centroids(cities, k_clusters)
    
    for _ in range(iterations):
        # Assign city to centroid
        assignments = assign_datum_to_cluster(cities, centroids)
        plot_cities(cities, assignments, centroids)
        # Termination Criteria
        # Update centroid
        new_centroids = get_cluster_mean(cities, assignments)
        if termination_criteria(centroids, new_centroids):
            break
        centroids = new_centroids
        
    
    return centroids, assignments

# Main
def run_k_means(k_clusters):
    k_clusters = 12 # days
    cities = read_cities()
    centroids, assignments = k_means(k_clusters=k_clusters, cities=cities, iterations=100)
    
    # Analyze
    clustered_cities = _gather_cities_by_cluster(cities=cities, assignments=assignments)
    for centroid_idx, cluster in clustered_cities.items():
        print(f'Centroid {centroid_idx} has {len(cluster)} cities')
        
    plot_cities(cities, assignments, centroids, autoclose=False)
    
    return centroids, clustered_cities

if __name__=='__main__':
    k_clusters = 12 # days
    centroids, clustered_cities = run_k_means(k_clusters=k_clusters)
        