import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim

# Initialize the geolocator
geolocator = Nominatim(user_agent="Pocket-Trains-Project")

def main():
    # Load your data into a DataFrame
    location_df = read_location_data("PocketTrainsWithLocations.csv")
    
    # Plot the map
    plot_map(location_df, "PocketTrainsMap.png")

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data into a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")

# Function to get location coordinates with rate limiting and caching
location_cache = {}

def get_location(city: str) -> tuple:
    """Get location coordinates for a given city."""
    if city in location_cache:
        print(f"Using cached coordinates for city: {city} - Coordinates: {location_cache[city]}")
        return location_cache[city]
    
    try:
        location = geolocator.geocode(city)
        if location:
            coordinates = (location.latitude, location.longitude)
            location_cache[city] = coordinates
            print(f"Processed city: {city} - Coordinates: {coordinates}")
            return coordinates
        else:
            print(f"Location not found for city: {city}")
            return (None, None)
    except Exception as e:
        print(f"Error occurred while fetching location for city: {city}. Error: {e}")
        return (None, None)

# Function to read the location data from the CSV file
def read_location_data(file_path: str) -> pd.DataFrame:
    """Read location data from a CSV file."""
    try:
        location_data = pd.read_csv(file_path)
        print(location_data.to_string())
        return location_data
    except Exception as e:
        print(f"Error occurred while reading the file: {file_path}. Error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by removing duplicates and invalid coordinates."""
    # Remove duplicates based on the 'Location' column
    original_count = len(df)
    df = df.drop_duplicates(subset='Location')
    removed_count = original_count - len(df)
    print(f"Removed {removed_count} duplicate entries.")

    # Filter out invalid coordinates
    df = df[df['Location'].apply(lambda loc: isinstance(loc, tuple) and 
                                  pd.notna(loc) and 
                                  -90 <= loc[0] <= 90 and 
                                  -180 < loc[1] < 180)]
    
    return df

def plot_map(df: pd.DataFrame, output_file: str) -> None:
    """Plot a map with locations from the DataFrame and save it as a PNG file."""
    # Clean the data
    df = clean_data(df)

    plt.figure(figsize=(10, 8))
    m = Basemap(projection='lcc', resolution='h', 
                lat_0=0, lon_0=0,  # Specify the center of the map
                lat_1=30, lat_2=60,  # Updated values for lat_1 and lat_2
                llcrnrlon=-180, llcrnrlat=-60, 
                urcrnrlon=180, urcrnrlat=60)
    m.drawcoastlines()
    m.drawcountries()

    # Plot each location
    for index, row in df.iterrows():
        lat, lon = row['Location']
        try:
            x, y = m(lon, lat)
            m.plot(x, y, 'bo', markersize=5)  # Plotting the location as a blue dot
        except Exception as e:
            print(f"Error plotting coordinates for index {index}: ({lat}, {lon}). Error: {e}")

    plt.title("Map of Locations")
    try:
        plt.savefig(output_file, format='png')
        plt.close()
        print(f"Map saved to {output_file}")
    except Exception as e:
        print(f"Error saving the map: {e}")

if __name__ == "__main__":
    main()
