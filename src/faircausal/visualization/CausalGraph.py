import matplotlib.pyplot as plt
import networkx as nx


def show_graph(graph_dict, title="Causal Graph", save_path=None, figsize=(10, 7),
               node_color='lightblue', edge_color='gray', edge_width=1, arrow_size=20):
    """
    Visualize the causal graph with customizable edge properties.

    :param graph_dict: Dictionary representation of the causal graph
    :param title: Title of the graph
    :param save_path: Path to save the figure
    :param figsize: Size of the figure as a tuple (width, height)
    :param node_color: Color of the nodes
    :param edge_color: Color of the edges
    :param edge_width: Width of the edges
    :param arrow_size: Size of the arrows
    """
    graph = nx.DiGraph()

    for node, neighbors in graph_dict.items():
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)

    for node in graph_dict:
        if node not in graph:
            graph.add_node(node)

    pos = nx.spring_layout(graph)

    plt.figure(figsize=figsize)
    nx.draw(graph, pos, with_labels=True, node_color=node_color,
            node_size=3000, font_size=16, font_weight='bold',
            arrows=True, arrowsize=arrow_size,
            edge_color=edge_color, width=edge_width)

    nx.draw_networkx_labels(graph, pos, font_size=16, font_weight="bold")

    plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)

    plt.show()
