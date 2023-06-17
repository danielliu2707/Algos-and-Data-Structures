from graph_class import Graph

def traverse(Graph_object: Graph):
    visited = [0] * Graph_object.no_vertices
    for i in range(Graph_object.no_vertices):
        # If vertex not visited, then call DFS on it
        if visited[i] == 0:
            # Don't actually need any visited = ... All same list being modified.
            visited = depth_first_search(Graph_object, i, visited)
    return visited  # Don't actually need to return bit_list


def depth_first_search(Graph_object: Graph, i: int, visited: list):
    """
        Implementing DFS

        Complexity: O(V + E)
    """
    visited[i] = 1
    no_adjacent = len(Graph_object.arr[i][1])
    for j in range(no_adjacent):
        Edge = Graph_object.arr[i][1][j]
        # The vertex this edge is connected to (i.e Adjacent to)
        v = Edge.connected_to
        if visited[v] == 0:
            visited = depth_first_search(Graph_object, v, visited)
    return visited   # This return here occurs when we start to "backtrack".