# Implementing a class for a Directed, Weighted Graph

class Edge:
    """
        A weighted, directed edge which defines a relationship TO v from some other vertex with weigh w.
    """

    def __init__(self, u, v, w):
        self.connected_from = u
        self.connected_to = v
        self.weight = w


class Graph:
    """
        A general Graph Class which works for either Directed/Undirected, Weighted/Unweighted Graph.

        By Default: Creates an Undirected, Unweighted Graph.

        If creating a:
        - Unweighted Graph: Must not provide a weight.
        - Weighted Graph: Must provide a weight
    """

    def __init__(self, n, directed=False, weighted=False):
        """
            Initializes graph with n vertices.
            Vertex IDs go from 0 to n-1 (inclusive).
        """
        self.directed = directed
        self.weighted = weighted
        self.no_vertices = n
        self.arr = [(v, []) for v in range(n)]

    def add_edge(self, u, v, w=None):
        """
            Adds weighted edge between (u, v) of weight w.
            This means u is directed to v (i.e. stored in A[u] is Edge(v))
        """
        # If Graph is weigted, and w is None, failed to provide weight argument
        if self.weighted is True:
            assert w is not None, "Failed to provide a weight when it's a weighted graph"
        # Incase user accidentally inserts a weight when graph is unweighted
        else:
            assert w is None, "Provided a weight when it's an unweighted graph"
        # Testing for whether directed or not
        if not self.directed:
            self.arr[u][1].append(Edge(u, v, w))
            self.arr[v][1].append(Edge(v, u, w))
        else:
            self.arr[u][1].append(Edge(u, v, w))


# Implementation for only a DW_Graph Class:
class DW_Graph:
    """
        A directed, weighted graph class. It only works for node values {0, 1, 2, ..., n}
    """

    def __init__(self, number_vertices):
        """
            Initializes graph with n vertices.
            Vertex IDs go from 0 to n-1 (inclusive).
        """
        self.arr = [(v, []) for v in range(number_vertices)]
        self.number_vertices = number_vertices

    def add_weighted_edge(self, u, v, w):
        """
            Adds weighted edge between (u, v) of weight w.
            This means u is directed to v (i.e. stored in A[u] is Edge(v))
        """
        self.arr[u][1].append(Edge(u, v, w))

    def get_adjacent_edges(self, u) -> list:
        """
            Retrieves all adjacent edges from a vertex u by returning a list of edges
        """
        return self.arr[u][1]

    def get_all_edges(self) -> list:
        """
            Retrieves all edges within a graph.

            Time Complexity: O(|E|).
        """
        lst_edges = []
        # Get all adjacent edges for each vertex. Go through this list of adjacent edges and append to overall lst_edges:
        for i in range(len(self.arr)):
            adjacent_edges = self.get_adjacent_edges(self.arr[i][0])
            for edge in adjacent_edges:
                lst_edges.append(edge)
        return lst_edges
