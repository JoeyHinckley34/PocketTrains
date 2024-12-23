import pandas as pd
import networkx as nx
import folium
from folium import plugins
import ast
import warnings
import os
import time


def process_train_network_data(file_path):
    """
    Process the train network data from CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Processed DataFrame with extracted coordinates
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting data processing...")
    start_time = time.time()
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
        
   
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Check for required columns
        required_columns = ['Source', 'Destination', 'Location', 'Fuel', 'Water']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None

        # Function to safely extract coordinates from location string
        def extract_coords(loc_str):
            try:
                if pd.isna(loc_str):
                    return None, None
                # Convert string tuple to actual tuple
                coords = ast.literal_eval(loc_str)
                return coords[0], coords[1]
            except Exception as e:
                print(f"Error parsing coordinates: {loc_str}. Error: {str(e)}")
                return None, None
        
        # Extract coordinates with error handling
        try:
            df[['source_lat', 'source_long']] = df.apply(
                lambda row: pd.Series(extract_coords(row['Location'])), axis=1
            )
            df[['dest_lat', 'dest_long']] = df.apply(
                lambda row: pd.Series(extract_coords(row['Location'])), axis=1
            )
        except Exception as e:
            print(f"Error extracting coordinates: {str(e)}")
            return None

        print(f"[{time.strftime('%H:%M:%S')}] Data processing completed in {time.time() - start_time:.2f} seconds")
        return df

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None

def create_train_network_map(df, routes=None):
    """
    Create an interactive map visualization of train networks using folium.
    
    Parameters:
    df (pandas.DataFrame): Processed DataFrame
    routes (list): List of routes for each train
    
    Returns:
    tuple: (folium.Map, networkx.Graph) Interactive map and network graph
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting map creation...")
    start_time = time.time()
    
    # Define colors for different train routes
    train_colors = ['#e74c3c', '#2ecc71', '#3498db', '#f1c40f', '#9b59b6', 
                    '#1abc9c', '#e67e22', '#34495e', '#7f8c8d', '#c0392b']

    if df is None or df.empty:
        print("Error: No valid data to create map")
        return None, None
        
    try:
        # Create a network graph
        G = nx.Graph()
        
        # Create a dictionary to store city coordinates
        city_coords = {}
        
        # First pass: collect all unique cities and their coordinates
        all_cities = pd.concat([
            df[['Source', 'source_lat', 'source_long']].rename(
                columns={'Source': 'city', 'source_lat': 'lat', 'source_long': 'long'}
            ),
            df[['Destination', 'dest_lat', 'dest_long']].rename(
                columns={'Destination': 'city', 'dest_lat': 'lat', 'dest_long': 'long'}
            )
        ]).drop_duplicates('city')
        
        # Add nodes (cities) to the graph
        for _, row in all_cities.iterrows():
            if pd.notna(row['lat']) and pd.notna(row['long']):
                city_coords[row['city']] = (row['lat'], row['long'])
                G.add_node(row['city'], pos=(row['lat'], row['long']))
        
        if not city_coords:
            print("Error: No valid coordinates found")
            return None, None

        # Add edges without costs
        for _, row in df.iterrows():
            if (row['Source'] in city_coords and 
                row['Destination'] in city_coords):
                G.add_edge(
                    row['Source'],
                    row['Destination'],
                    fuel=row['Fuel'],  # Only keep fuel
                    water=row['Water']  # Keep water if needed
                )
        
        # Calculate center of the map
        valid_coords = [(lat, long) for lat, long in city_coords.values() 
                       if pd.notna(lat) and pd.notna(long)]
        center_lat = sum(coord[0] for coord in valid_coords) / len(valid_coords)
        center_long = sum(coord[1] for coord in valid_coords) / len(valid_coords)
        
        # Create the base map
        m = folium.Map(location=[center_lat, center_long], 
                      zoom_start=4,
                      tiles='CartoDB positron')
        
        # Add cities as markers with coordinates in the popup
        for city, coords in city_coords.items():
            folium.CircleMarker(
                location=coords,
                radius=6,
                popup=f"{city}<br>Coordinates: {coords[0]}, {coords[1]}",
                color='#3186cc',
                fill=True,
                fill_color='#3186cc',
                fill_opacity=0.7
            ).add_to(m)
        
        # Add train connections as lines
        if routes:
            # Create a dictionary to track which edges belong to which train
            edge_colors = {}
            for train_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    city1, city2 = route[i], route[i + 1]
                    edge = tuple(sorted([city1, city2]))  # Sort to ensure consistent edge identification
                    edge_colors[edge] = train_colors[train_idx % len(train_colors)]
            
            # Draw the edges with their assigned colors
            for (city1, city2, data) in G.edges(data=True):
                coords1 = city_coords[city1]
                coords2 = city_coords[city2]
                edge = tuple(sorted([city1, city2]))
                color = edge_colors.get(edge, '#808080')  # Use gray for unassigned edges
                
                popup_content = f"""
                <b>{city1} to {city2}</b><br>
                Fuel: {data['fuel']}<br>
                Water: {data['water']}
                """
                
                folium.PolyLine(
                    locations=[coords1, coords2],
                    weight=2,
                    color=color,
                    popup=popup_content,
                    opacity=0.8
                ).add_to(m)
        else:
            # Original code for when no routes are provided
            for (city1, city2, data) in G.edges(data=True):
                coords1 = city_coords[city1]
                coords2 = city_coords[city2]
                
                popup_content = f"""
                <b>{city1} to {city2}</b><br>
                Fuel: {data['fuel']}<br>
                Water: {data['water']}
                """
                
                folium.PolyLine(
                    locations=[coords1, coords2],
                    weight=2,
                    color='#e74c3c',
                    popup=popup_content,
                    opacity=0.8
                ).add_to(m)

        # Add a fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Add a legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white;
                    padding: 10px;
                    opacity: 0.8;">
            <p style="margin: 0"><span style="color: #3186cc;">●</span> Cities</p>
        '''
        if routes:
            for i, color in enumerate(train_colors[:len(routes)]):
                legend_html += f'<p style="margin: 0"><span style="color: {color};">―</span> Train {i+1}</p>'
        else:
            legend_html += '<p style="margin: 0"><span style="color: #e74c3c;">―</span> Train Routes</p>'
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        print(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        print(f"Collected coordinates for {len(city_coords)} cities")
        
        print(f"[{time.strftime('%H:%M:%S')}] Map creation completed in {time.time() - start_time:.2f} seconds")
        return m, G  # Return both the map and the graph

    except Exception as e:
        print(f"Error creating map: {str(e)}")
        return None, None

def dfs_train_network(G, start_city):
    """
    Perform Depth-First Search on the train network.
    
    Parameters:
    G (networkx.Graph): The train network graph
    start_city (str): Starting city for DFS
    
    Returns:
    list: List of cities in DFS order
    """
    if start_city not in G:
        print(f"Error: Start city '{start_city}' not found in network")
        return None
        
    visited = []
    
    def dfs_recursive(city):
        visited.append(city)
        for neighbor in G[city]:
            if neighbor not in visited:
                dfs_recursive(neighbor)
    
    dfs_recursive(start_city)
    return visited

def assign_train_routes(G, num_trains):
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting route assignment...")
    start_time = time.time()
    
    if num_trains <= 0:
        print("Error: Number of trains must be positive")
        return None
        
    if num_trains > len(G.nodes):
        print("Error: More trains than cities")
        return None
    
    routes = [[] for _ in range(num_trains)]
    route_costs = [0] * num_trains  # Track total fuel cost of each route
    unassigned_cities = set(G.nodes())
    
    # Start with major hub cities (highest degree) for each train
    hub_cities = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:num_trains]
    for i, city in enumerate(hub_cities):
        routes[i].append(city)
        unassigned_cities.remove(city)
    
    while unassigned_cities:
        # Find the route with the lowest total fuel cost
        cheapest_route_idx = min(range(num_trains), key=lambda i: route_costs[i])
        current_route = routes[cheapest_route_idx]
        current_city = current_route[-1]
        
        # Find all unassigned neighbors
        neighbors = [n for n in G.neighbors(current_city) if n in unassigned_cities]
        
        if neighbors:
            # Choose the neighbor with lowest fuel cost
            next_city = min(neighbors, 
                          key=lambda n: G[current_city][n]['fuel'])
            current_route.append(next_city)
            route_costs[cheapest_route_idx] += G[current_city][next_city]['fuel']
            unassigned_cities.remove(next_city)
        else:
            # If no unassigned neighbors, find the closest unassigned city to any city in the route
            if unassigned_cities:
                min_fuel = float('inf')
                best_pair = None
                
                for route_city in current_route:
                    for unassigned in unassigned_cities:
                        if G.has_edge(route_city, unassigned):
                            fuel_cost = G[route_city][unassigned]['fuel']
                            if fuel_cost < min_fuel:
                                min_fuel = fuel_cost
                                best_pair = (route_city, unassigned)
                
                if best_pair:
                    next_city = best_pair[1]
                    current_route.append(next_city)
                    route_costs[cheapest_route_idx] += min_fuel
                    unassigned_cities.remove(next_city)
                else:
                    # If no connection found, start a new segment with the highest-degree unassigned city
                    next_city = max(unassigned_cities, key=lambda c: G.degree(c))
                    current_route.append(next_city)
                    unassigned_cities.remove(next_city)
    
    # Print route statistics
    print("\nRoute Statistics:")
    for i, (route, cost) in enumerate(zip(routes, route_costs), 1):
        print(f"Train {i} ({len(route)} cities, Total fuel: {cost}): {' -> '.join(route)}")
    
    print(f"[{time.strftime('%H:%M:%S')}] Route assignment completed in {time.time() - start_time:.2f} seconds")
    return routes

def load_and_process_data(file_path='PocketTrainsWithLocations.csv'):
    """Load and process the train network data from CSV."""
    print(f"Processing data file: '{file_path}'")
    df = process_train_network_data(file_path)
    
    if df is not None:
        print(f"Loaded {len(df)} train connections")
        return df
    return None

def create_initial_network(df):
    """Create the initial network map and graph."""
    train_network_map, train_graph = create_train_network_map(df)
    return train_network_map, train_graph

def generate_train_routes(train_graph, num_trains=10):
    """Generate and display train routes."""
    print(f"\nAssigning {num_trains} trains to routes...")
    routes = assign_train_routes(train_graph, num_trains)
    
    if routes:
        for i, route in enumerate(routes, 1):
            if route:
                print(f"Train {i} ({len(route)} cities): {' -> '.join(route)}")
            else:
                print(f"Train {i}: No cities assigned")
    return routes

def save_network_map(train_network_map, filename='train_network.html'):
    """Save the network map to an HTML file."""
    print(f"\nSaving map to '{filename}'...")
    train_network_map.save(filename)

def process_train_network(file_path='PocketTrainsWithLocations.csv', num_trains=10):
    """
    Process the entire train network workflow from data loading to map creation.
    Returns the final map and graph, or raises an exception if any step fails.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting program...")
    total_start_time = time.time()
    
    try:
        df = process_train_network_data(file_path)
        if df is None:
            raise ValueError("Failed to process train network data")
            
        _, train_graph = create_train_network_map(df)
        if train_graph is None:
            raise ValueError("Failed to create initial network")
            
        routes = assign_train_routes(train_graph, num_trains)
        if routes is None:
            raise ValueError("Failed to generate train routes")
            
        train_network_map, _ = create_train_network_map(df, routes)
        if train_network_map is None:
            raise ValueError("Failed to create final network map")
            
        save_network_map(train_network_map)
        
        print(f"\n[{time.strftime('%H:%M:%S')}] Program completed in {time.time() - total_start_time:.2f} seconds")
        return train_network_map, train_graph
        
    except Exception as e:
        print(f"Error processing train network: {str(e)}")
        return None, None

if __name__ == '__main__':
    process_train_network()

