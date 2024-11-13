import pandas as pd
import networkx as nx
import folium
from folium import plugins
import ast

def process_train_network_data(file_path):
    """
    Process the train network data from CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Processed DataFrame with extracted coordinates
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Function to safely extract coordinates from location string
    def extract_coords(loc_str):
        try:
            if pd.isna(loc_str):
                return None, None
            # Convert string tuple to actual tuple
            coords = ast.literal_eval(loc_str)
            return coords[0], coords[1]
        except:
            return None, None
    
    # Extract coordinates for each location
    df[['source_lat', 'source_long']] = df.apply(
        lambda row: pd.Series(extract_coords(row['Location'])), axis=1
    )
    
    return df

def create_train_network_map(df, cost_column='Cost to Build'):
    """
    Create an interactive map visualization of train networks using folium.
    
    Parameters:
    df (pandas.DataFrame): Processed DataFrame
    cost_column (str): Which cost column to use for edge weights
    
    Returns:
    folium.Map: Interactive map with train network
    """
    # Create a network graph
    G = nx.Graph()
    
    # Create a dictionary to store city coordinates
    city_coords = {}
    
    # First pass: collect all unique cities and their coordinates
    all_cities = pd.concat([
        df[['Source', 'source_lat', 'source_long']].rename(
            columns={'Source': 'city', 'source_lat': 'lat', 'source_long': 'long'}
        ),
        df[['Destination', 'source_lat', 'source_long']].rename(
            columns={'Destination': 'city', 'source_lat': 'lat', 'source_long': 'long'}
        )
    ]).drop_duplicates('city')
    
    # Add nodes (cities) to the graph
    for _, row in all_cities.iterrows():
        if pd.notna(row['lat']) and pd.notna(row['long']):
            # Invert coordinates for "Broome"
            if row['city'] == "Broome":
                city_coords[row['city']] = (row['long'], row['lat'])  # Inverted
            else:
                city_coords[row['city']] = (row['lat'], row['long'])
            G.add_node(row['city'], pos=(row['lat'], row['long']))
    
    # Add edges with costs
    for _, row in df.iterrows():
        if (row['Source'] in city_coords and 
            row['Destination'] in city_coords):
            G.add_edge(
                row['Source'],
                row['Destination'],
                weight=row[cost_column],
                water=row['Water'],
                fuel=row['Fuel']
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
    
    # Add cities as markers
    for city, coords in city_coords.items():
        folium.CircleMarker(
            location=coords,
            radius=6,
            popup=city,
            color='#3186cc',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.7
        ).add_to(m)
    
    # Add train connections as lines
    for (city1, city2, data) in G.edges(data=True):
        coords1 = city_coords[city1]
        coords2 = city_coords[city2]
        
        # Create popup content with all relevant information
        popup_content = f"""
        <b>{city1} to {city2}</b><br>
        Cost: {data['weight']}<br>
        Fuel: {data['fuel']}<br>
        Water: {data['water']}
        """
        
        # Create a line for the train connection
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
        <p style="margin: 0"><span style="color: #e74c3c;">―</span> Train Routes</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# Example usage:

# Process the data
df = process_train_network_data('PocketTrainsWithLocations.csv')

# Create the map using different cost columns
build_cost_map = create_train_network_map(df, cost_column='Cost to Build')
build_cost_map.save('train_network_build_costs.html')

claim_cost_map = create_train_network_map(df, cost_column='Cost to Claim')
claim_cost_map.save('train_network_claim_costs.html')
