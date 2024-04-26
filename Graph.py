import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
G.add_nodes_from(["Start", "Quantitative Analysis", "Qualitative Research Methods", "Experimental Design", "Interrelationships", "End"])

# Add edges
edges = [("Start", "Quantitative Analysis"),
         ("Start", "Qualitative Research Methods"),
         ("Start", "Experimental Design"),
         ("Quantitative Analysis", "Interrelationships"),
         ("Qualitative Research Methods", "Interrelationships"),
         ("Experimental Design", "Interrelationships"),
         ("Interrelationships", "End")]

G.add_edges_from(edges)

# Set node positions
pos = {"Start": (1, 3),
       "Quantitative Analysis": (0, 2),
       "Qualitative Research Methods": (1, 2),
       "Experimental Design": (2, 2),
       "Interrelationships": (1, 1),
       "End": (1, 0)}

# Draw nodes and edges
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
plt.title("Research Methodologies Flowchart")
plt.show()
