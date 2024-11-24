import pandas as pd
import networkx as nx
import folium
from folium import plugins
import ast
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)

def process_train_network_data(file_path):
    """
    Process the train network data from CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Processed DataFrame with extracted coordinates
    """
    print("1")
   
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
        
    print("2")
   
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        print("3")
        # Add diagnostic print
        print("Available columns:", df.columns.tolist())
        print("4")
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

        return df

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None

def create_train_network_map(df):
    """
    Create an interactive map visualization of train networks using folium.
    
    Parameters:
    df (pandas.DataFrame): Processed DataFrame
    
    Returns:
    folium.Map: Interactive map with train network
    """
    if df is None or df.empty:
        print("Error: No valid data to create map")
        return None
        
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
            return None

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
        for (city1, city2, data) in G.edges(data=True):
            coords1 = city_coords[city1]
            coords2 = city_coords[city2]
            
            # Create popup content with relevant information
            popup_content = f"""
            <b>{city1} to {city2}</b><br>
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

    except Exception as e:
        print(f"Error creating map: {str(e)}")
        return None

# Example usage:

# Process the data
df = process_train_network_data('PocketTrainsWithLocations.csv')

# # Create the map
train_network_map = create_train_network_map(df)
train_network_map.save('train_network.html')

