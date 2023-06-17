# Bellman-Ford: Single source shortest path with negative weights:

from graph_class import DW_Graph


def bellman_ford(graph: DW_Graph, source: int) -> list:
    """
        The following algorithm finds the shortest path from a source to all other vertices with negative weights.

        Approach:
        - Initialize distances to all be infinity except source to be 0.
        - Go through |V| - 1 iterations, and in each iteration, go through all edges and attempt to "relax".
        - Relaxation means using the recurisve formula to update values in memo (dist): min(dist[v], dist[u] + w(u, v)).

        Two guarantees:
        1. If there's no negative cycles, it'll return the shortest distance from some source s to every other vertex v.
        2. If there is a negative cycle, it'll detect it and return an error.

        Time Complexity: O(VE)
        Aux Space Complexity: O(V) as size of dist_array is |V|
    """
    # First Guarantee:
    number_vertices = graph.number_vertices
    dist_array = [float("inf")] * number_vertices    # My Memo array.
    pred_array = [None] * number_vertices
    dist_array[source] = 0
    # Iterate |V| - 1 times: To produce the optimal simple path
    for i in range(number_vertices - 1):
        lst_edges = graph.get_all_edges()
        # Iterate through |E| edges and attempt to relax:
        for edge in lst_edges:
            vertex_from = edge.connected_from
            vertex_to = edge.connected_to
            weight = edge.weight
            # Attempt to relax:
            if dist_array[vertex_to] > dist_array[vertex_from] + weight:
                dist_array[vertex_to] = dist_array[vertex_from] + weight
                pred_array[vertex_to] = vertex_from

    # Second Guarantee: Going |V|th iteration to check if there is a negative cycle:
    lst_edges = graph.get_all_edges()
    # Iterate through |E| edges and attempt to relax:
    for edge in lst_edges:
        vertex_from = edge.connected_from
        vertex_to = edge.connected_to
        weight = edge.weight
        # Attempt to relax: If any relaxation occurs in |V|th iteration, throw an error.
        if dist_array[vertex_to] > dist_array[vertex_from] + weight:
            raise Exception("Negative Cycle")

    return (dist_array, pred_array)
