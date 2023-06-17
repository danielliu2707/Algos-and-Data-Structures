# Implementation of Floyd-Warshall

from graph_class import DW_Graph


def floyd_warshall(graph: DW_Graph) -> list:
    """
        This algorithm returns an adjacency matrix containing the minimum distance from every vertex u to v.
        It solves the all-pairs, shortest path problem.

        Approach:
        - Initialize adjacency matrix to all infinity.
        - Initialize diagonals to be 0 and all pre-existing edges to their weights.
        - For each intermediary node,
            - For each pair of vertices (u, v),
                - Set the distance to the min(current distance, distance whilst going through intermediary node k)

        Guarantees:
        1. If there is no negative cycles, after the k'th iteration, dist[u][v] will contain
        the distance/weight of the shortest path from vertex u to v, potentially using intermediary nodes from set {1..k}
        2. If there are negative cycles, they will always be detected.
    """
    number_vertices = graph.number_vertices
    dist_arr = [[float("inf") for _ in range(number_vertices)]
                for _ in range(number_vertices)]   # Initialize 2D Memo: O(V^2)
    # Initialize diagonals to be 0: O(V)
    for i in range(number_vertices):
        dist_arr[i][i] = 0

    # Initialize pre-existing edges: O(E)
    lst_edges = graph.get_all_edges()
    for edge in lst_edges:
        # Getting attributes of edge
        u = edge.connected_from
        v = edge.connected_to
        weight = edge.weight
        # Setting the edge weight in 2D Memo
        dist_arr[u][v] = weight

    # Updating distances using intermediary nodes: O(V^3):
    for k in range(number_vertices):
        for u in range(number_vertices):
            for v in range(number_vertices):
                dist_arr[u][v] = min(
                    dist_arr[u][v], dist_arr[u][k] + dist_arr[k][v])

    # If there are any negatives along diagonal, we know negative cycle:
    # This means that all nodes reachable from the negative cycle can
    # continuously have their distances updated, and thus, will technically be wrong
    for i in range(number_vertices):
        if dist_arr[i][i] < 0:
            raise Exception("Negative Cycle")

    # Return array containing min distance for all vertex pairs:
    return dist_arr