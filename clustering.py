'''
    Note: Each city has an unique location (lat, lon)
'''
# Type Hints
from typing import List, Tuple, Dict
# Plot
import matplotlib.pyplot as plt
import folium
# Main
from helpers import calculate_distance, read_cities
from collections import defaultdict
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
    cities (List[Tuple[float, float]]): (numpy Array) A list of tuples where each tuple represents a geographic city (latitude, longitude).
    k (int): The number of random cities to select.

    Returns:
    List[float, float]: (numpy Array) A list of k randomly selected cities.
    """
    if k > len(cities):
        raise ValueError("k cannot be greater than the number of cities in the list")
    return cities[np.random.choice(cities.shape[0], k, replace=False)]#random.sample(cities, k) 

# Assigment Step
def assign_datum_to_cluster(
    cities: List[Tuple[float, float]], 
    centroids: List[Tuple[float, float]]) -> List[int]:
    """
    Assign each city to the nearest centroid by calculating the geodesic distance to each centroid.
    A city is automatically assigned to itself if it is also a centroid.

    Args:
    cities (List[Tuple[float, float]]): A list of tuples, where each tuple contains coordinates of a city.
    centroids (List[Tuple[float, float]]): A list of tuples, where each tuple contains coordinates of a centroid.

    Returns:
    List[int]: A list where the index represents the index of a city in the cities input list, and the value at that index
               represents the index of the closest centroid in the centroids input list.
    """
    centroid_set = {tuple(centroid): idx for idx, centroid in enumerate(centroids)}
    assignments = []

    for city in cities:
        if tuple(city) in centroid_set:
            assignments.append(centroid_set[tuple(city)])
        else:
            distances = [calculate_distance(city, centroid) for centroid in centroids]
            assigned_centroid = np.argmin(distances)
            assignments.append(assigned_centroid)

    return assignments

# Update Step
def gather_cities_by_centroid(cities: List[Tuple[float, float]], assignments: List[int]) -> Dict[int, List[Tuple[float, float]]]:
    """
    Group cities by their assigned centroid.

    Args:
    cities (List[Tuple[float, float]]): A list of tuples, where each tuple contains the coordinates of a city.
    assignments (List[int]): A list of integers where each value represents the index of the centroid 
                             assigned to the corresponding city in the cities list.

    Returns:
    Dict[int, List[Tuple[float, float]]]: A dictionary where the keys are centroid indices and the values 
                                          are lists of cities assigned to each centroid.
    """
    clusters = defaultdict(list)

    for city_index, cluster_index in enumerate(assignments):
        clusters[cluster_index].append(cities[city_index])

    return dict(clusters)

def get_cluster_mean(cities: List[Tuple[float, float]], assignments: List[int]) -> List[Tuple[float, float]]:
    """
    Calculate the mean coordinates of each cluster.

    Args:
    cities (List[Tuple[float, float]]): A list of tuples, where each tuple contains the coordinates of a city.
    assignments (List[int]): A list of integers where each value represents the index of the centroid 
                             assigned to the corresponding city in the cities list.

    Returns:
    List[Tuple[float, float]]: A list of tuples, where each tuple contains the mean coordinates of a cluster.
    """
    clusters = gather_cities_by_centroid(cities, assignments)
    new_centroids = [
        ( np.mean([lon for lon, _ in clustered_cities]),
         np.mean([lat for _, lat in clustered_cities]) )
        for clustered_cities in clusters.values()
    ]
    return new_centroids

# Termination Criteria
def termination_criteria(
    old_centroids: List[Tuple[float, float]], 
    new_centroids: List[Tuple[float, float]], 
    tolerance: float = 1e-3) -> bool:
    """
    Determine if the termination criteria for centroid convergence is met.

    Args:
    old_centroids (List[Tuple[float, float]]): The list of old centroid coordinates.
    new_centroids (List[Tuple[float, float]]): The list of new centroid coordinates.
    tolerance (float): The tolerance threshold to determine convergence. 1meter

    Returns:
    bool: True if the maximum difference between old and new centroids is less than the tolerance, False otherwise.
    """
    max_difference = max(
        calculate_distance(old_centroid, new_centroids[i]) 
        for i, old_centroid in enumerate(old_centroids)
    )
    
    return max_difference < tolerance
        
# Clustering
def k_means(
    k_clusters: int, 
    cities: List[Tuple[float, float]], 
    iterations: int) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Perform K-means clustering.

    Args:
    k_clusters (int): Number of clusters.
    cities (List[Tuple[float, float]]): A list of tuples, where each tuple contains the coordinates of a city.
    iterations (int): Maximum number of iterations.

    Returns:
    Tuple[List[Tuple[float, float]], List[int]]: Final centroids and city assignments.
    """
    centroids = initialize_centroids(cities, k_clusters)

    for _ in range(iterations):
        # Assign city to centroid
        assignments = assign_datum_to_cluster(cities, centroids)
        
        # Plot the cities and centroids (optional, can be commented out for performance)
        plot_cities(cities, assignments, centroids)
        
        # Update centroid
        new_centroids = get_cluster_mean(cities, assignments)
        
        # Check termination criteria
        if termination_criteria(centroids, new_centroids):
            break
        
        centroids = new_centroids

    return centroids, assignments

# Main
def run_k_means(k_clusters: int) -> Tuple[List[Tuple[float, float]], Dict[int, List[Tuple[float, float]]]]:
    """
    Run K-means clustering on city data.

    Args:
    k_clusters (int): Number of clusters.

    Returns:
    Tuple[List[Tuple[float, float]], Dict[int, List[Tuple[float, float]]]]: Final centroids and clustered cities.
    """
    cities = read_cities()
    centroids, assignments = k_means(k_clusters=k_clusters, cities=cities, iterations=100)
    
    # Analyze
    clustered_cities = gather_cities_by_centroid(cities=cities, assignments=assignments)
    for centroid_idx, cluster in clustered_cities.items():
        print(f'Centroid {centroid_idx} has {len(cluster)} cities')
        
    plot_cities(cities, assignments, centroids, autoclose=False)
    
    return centroids, clustered_cities

if __name__=='__main__':
    k_clusters = 6 # days
    centroids, clustered_cities = run_k_means(k_clusters=k_clusters)
        