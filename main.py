import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the CSV into a pandas DataFrame
df = pd.read_csv("PocketTrainsEdges.csv")

# Initialize an empty graph
G = nx.Graph()

# Add edges to the graph with weights (fuel or cost to build, for example)
for index, row in df.iterrows():
    source = row['Source']
    destination = row['Destination']
    fuel = row['Fuel'] 
    G.add_edge(source, destination, weight=fuel)

print(G)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=8)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.show()
