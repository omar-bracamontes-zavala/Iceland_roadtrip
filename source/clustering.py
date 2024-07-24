'''
    Note: Each city has an unique location (lat, lon)
'''
# Type Hints
from typing import List, Tuple, Dict
# Plot
import folium
import matplotlib.pyplot as plt
# Main
import numpy as np
import random, json
from geopy.distance import geodesic
from collections import defaultdict
from helpers import read_cities, load_distance_matrix_from_json, _save_file_as_json


#
# Plots
#
def plot_cities(cities: List[Tuple[float, float]], assignments: List[int], centroids: List[Tuple[float, float]], map_or_scatter: str='scatter', autoclose: bool=True):
    if map_or_scatter == 'scatter':
        # Unpacking the list of tuples into x and y coordinates
        lat, lon = zip(*cities) 
        
        # Find the number of unique clusters
        num_clusters = len(set(assignments))
        
        # Generate a colormap with enough colors for the clusters
        cmap = plt.get_cmap('tab20', num_clusters)
        
        # Creating the scatter plot with cluster-based colors
        scatter = plt.scatter(lon, lat, c=assignments, cmap=cmap, s=30, edgecolor='k', alpha=0.6)

        # Plotting the centroids with full opacity and same color as their cluster
        for i, (lat, lon) in enumerate(centroids):
            plt.scatter(lon, lat, c=[i], cmap=cmap, alpha=1, s=150, marker='x')

        # Adding title and labels
        plt.title('Iceland')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
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
# Tweaks (these are metrics to integrate real distance and time constrains to each cluster)
#
def _calculate_city_errors(
    cities: List[Tuple[float, float]], 
    assignments: List[int], 
    distance_matrix: List[List[Tuple[float, float]]], 
    T: int,
    print_info:bool = True,
) -> Dict[int, Dict]:
    """
    Calculate city errors based on given distances and times.
    
    Parameters:
    cities (List[Tuple[float, float]]): List of city coordinates.
    assignments (List[int]): List of cluster assignments for each city.
    distance_matrix (List[List[Tuple[float, float]]]): Matrix containing distances and times between cities.
    T (int): Driving time constraint per cluster in minutes.
    
    Returns:
    Dict[int, Dict]: Cluster metrics including average errors.
    """
    
    # Gather cities by their centroid assignments
    clusters_metrics = gather_cities_by_centroid(
        cities=cities, assignments=assignments, gather_coordinates_or_index='index'
    )
    
    for cluster_details in clusters_metrics.values():
        cluster_cities = cluster_details['cities']
        N = len(cluster_cities)
        t = T / N
        
        distances = np.zeros((N, N))
        times = np.zeros((N, N))
        
        for i, city_i in enumerate(cluster_cities):
            for j, city_j in enumerate(cluster_cities):
                if i != j:
                    dist, time = distance_matrix[city_i][city_j][0], distance_matrix[city_i][city_j][1]
                    distances[i, j] = dist if not dist else t # Rough approach to deal with NaNs
                    times[i, j] = time if not time else t
                            
        # print(f'\tDistance: {distances} \tTime: {times}')
        
        avg_distances = np.sum(distances, axis=1) / (N - 1)
        avg_times = np.sum(times, axis=1) / (N - 1)
        
        # if print_info:
        #     print(f'Avg Distance: {avg_distances} \tAverage Time: {avg_times}')
        
        cluster_details['avg_errors'] = list(avg_distances * avg_times / t)
        
    for cluster_details in clusters_metrics.values():
        cluster_details['error'] = np.mean(cluster_details['avg_errors'])
        cluster_details['stdev'] = np.std(cluster_details['avg_errors'])
    
    return clusters_metrics

def _assign_outlier_to_new_cluster(
    city: Tuple[float, float], 
    centroids: List[Tuple[float, float]],
    avoid_list: List[int] = []) -> List[int]:
    """
    Assign outlier city to the nearest centroid by calculating the geodesic distance to each centroid.
    Args:
    city (Tuple[float, float]): Coordinates of a city.
    centroids (List[Tuple[float, float]]): A list of tuples, where each tuple contains coordinates of a centroid.
    avoid_list (List[int]): A list with the cluster to avoid on assigment

    Returns:
    List[int]: A list where the index represents the index of a city in the cities input list, and the value at that index
               represents the index of the closest centroid in the centroids input list.
    """
    centroid_set = {tuple(centroid): idx for idx, centroid in enumerate(centroids)}

    distances = [geodesic(city, centroid).km for centroid in centroids]
    assigned_centroid = np.argmin(distances)
    
    if assigned_centroid in avoid_list:
        distances[assigned_centroid] = np.inf
        assigned_centroid = np.argmin(distances)
        
    return assigned_centroid

def _evaluate_clusters(
    cities: List[Tuple[float, float]], 
    assignments: List[int], 
    centroids: List[Tuple[float, float]],
    distance_matrix: List[List[Tuple[float, float]]], 
    T: int
    ):
    clusters_metrics = _calculate_city_errors(cities=cities, assignments=assignments, distance_matrix=distance_matrix, T=T)
        
    for cluster, cluster_details in clusters_metrics.items():
        threshold = cluster_details['error'] + 2 * cluster_details['stdev']
        filter_indexes = [i for i, avg_error in enumerate(cluster_details['avg_errors']) if avg_error > threshold]
        outliers = [cluster_details['cities'][i] for i in filter_indexes]
        
        # assign new clusters
        for outlier in outliers:
            assignments[outlier] = _assign_outlier_to_new_cluster(cities[outlier], centroids, list(assignments[outlier]))

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
    List[float, float]: (numpy Array) A list of k randomly selected (list index) cities.
    """
    if k > len(cities):
        raise ValueError("k cannot be greater than the number of cities in the list")
    return cities[np.random.choice(cities.shape[0], k, replace=False)]#random.sample(cities, k) 

# Assigment Step
def assign_cities_to_cluster(
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
            distances = [geodesic(city, centroid).km for centroid in centroids]
            assigned_centroid = np.argmin(distances)                
            assignments.append(assigned_centroid)

    return assignments

# Update Step
def gather_cities_by_centroid(cities: List[Tuple[float, float]], assignments: List[int], gather_coordinates_or_index:str = 'coordinates') -> Dict[int, List[Tuple[float, float]]]:
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
    if gather_coordinates_or_index == 'coordinates':
        clusters = defaultdict(list)

        for city_index, cluster_index in enumerate(assignments):
            clusters[int(cluster_index)].append(list(cities[city_index])) #int & list bc it will be json serializable

        return dict(clusters)

    elif gather_coordinates_or_index == 'index':

        clusters_metrics = defaultdict(lambda: {'cities': [], 'avg_errors': [], 'error': None, 'stdev': None})

        # Similar as gather_cities_by_centroid but instead of (lat_,lon) is by  city index to use them in distance matrix
        for city_index, cluster_index in enumerate(assignments):
            clusters_metrics[int(cluster_index)]['cities'].append(int(city_index))
        
        return dict(clusters_metrics)
    

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
        geodesic(old_centroid, new_centroids[i]).km 
        for i, old_centroid in enumerate(old_centroids)
    )
    
    return max_difference < tolerance
        
# Clustering
def k_means(
    k_clusters: int, 
    cities: List[Tuple[float, float]], 
    iterations: int,
    distance_matrix: List[List[Tuple[float, float]]], 
    T: int) -> Tuple[List[Tuple[float, float]], List[int]]:
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
        assignments = assign_cities_to_cluster(cities, centroids)
        
        # Analyze assigments
        _evaluate_clusters(cities, assignments, centroids, distance_matrix, T)
        
        # Plot the cities and centroids (optional, can be commented out for performance)
        # plot_cities(cities, assignments, centroids)
        
        # Update centroid
        new_centroids = get_cluster_mean(cities, assignments)
        
        # Check termination criteria
        if termination_criteria(centroids, new_centroids):
            break
        
        centroids = new_centroids

    return centroids, assignments

# Main
def run_k_means(k_clusters: int, T:int, cities_filepath: str = 'source/datasets/cities.csv') -> Tuple[List[Tuple[float, float]], Dict[int, List[Tuple[float, float]]]]:
    """
    Run K-means clustering on city data.

    Args:
    k_clusters (int): Number of clusters.
    T: driving time constrain per cluster. In minutes

    Returns:
    Tuple[List[Tuple[float, float]], Dict[int, List[Tuple[float, float]]]]: Final centroids and clustered cities.
    """
    cities = read_cities(cities_filepath)
    distance_matrix = load_distance_matrix_from_json()
    centroids, assignments = k_means(k_clusters=k_clusters, cities=cities, iterations=100, distance_matrix=distance_matrix, T=T)
    
    # Analyze
    clustered_cities = gather_cities_by_centroid(cities=cities, assignments=assignments)
    for centroid_idx, cluster in clustered_cities.items():
        print(f'Centroid {centroid_idx} has {len(cluster)} cities')
        
    clusters_metrics = _calculate_city_errors(cities=cities, assignments=assignments, distance_matrix=distance_matrix, T=T)
        
    plot_cities(cities, assignments, centroids, autoclose=False)
    
    return centroids, clustered_cities, clusters_metrics, assignments

if __name__=='__main__':
    k_clusters = 5 # days
    T_hours = 5
    centroids, clustered_cities, clusters_metrics, assignments = run_k_means(k_clusters=k_clusters, T=T_hours*60)
    
    # print(clusters_metrics)
    _save_file_as_json(
        {
            'centroids':centroids,
            'clustered_cities':json.dumps(clustered_cities),
            'clusters_metrics':json.dumps(clusters_metrics),
            'assignments':assignments
        },
        'clustered_cities.json'
    )

        
    print(clustered_cities, '\n', clusters_metrics)
        