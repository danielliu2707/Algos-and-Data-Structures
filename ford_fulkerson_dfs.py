# Ford-Fulkerson Implementation using DFS:

class Edge:
    """
        A weighted, directed edge which defines a relationship TO v from some other vertex with weigh w.
    """

    def __init__(self, u, v, f, c):
        self.connected_from = u
        self.connected_to = v
        self.flow = f
        self.capacity = c

    def add_reference_reverse_edge(self, reverse_edge):
        """
            Adds a reference to the reverse edge for a given edge.
        """
        self.reverse_edge = reverse_edge


class DW_Graph:
    """
        A directed, weighted graph class modified to account for flow, capacity and demand
    """

    def __init__(self, number_vertices, lst_demands: list):
        """
            Initializes graph with n vertices.
            Vertex IDs go from 0 to n-1 (inclusive).
        """
        self.arr = [(v, []) for v in range(number_vertices)]
        self.forward_edges = [(v, []) for v in range(number_vertices)]    # Array containing only forward edges
        self.number_vertices = number_vertices
        self.demands = lst_demands

    def add_weighted_edge(self, u, v, f, c):
        """
            Adds weighted edge between (u, v) of weight w.
            This means u is directed to v (i.e. stored in A[u] is Edge(v)).
            Initialize the corresponding reverse edge
        """
        forward_edge = Edge(u, v, f, c)
        backwards_edge = Edge(v, u, 0, 0)   # Initialize flow = 0, capacity = 0. Flow of backwards edges (i.e. flow to be redirected) is negative
        forward_edge.add_reference_reverse_edge(backwards_edge)
        backwards_edge.add_reference_reverse_edge(forward_edge)
        self.arr[u][1].append(forward_edge)
        self.arr[v][1].append(backwards_edge)   # Adding backwards edge to list of edges


    def get_adjacent_edges(self, u) -> list:
        """
            Retrieves all adjacent edges from a vertex u by returning a list of edges.
            Modified to not only include outgoing edges for u, but also
            Incoming edges for u, but with the reverse flow.
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

    def get_all_demands(self, only_positive = False, only_negative = False) -> list:
        """
            Returns a list of all demand values, allowing optional arguments to get only positive or negative demands.
            If both optinal arguments are False, gets all demands.
        """
        lst_demands = []
        for i in range(len(self.arr)):
            if only_positive:
                if self.arr[i][2] > 0:
                    lst_demands.append(self.arr[i][2])
            elif only_negative:
                if self.arr[i][2] < 0:
                    lst_demands.append(self.arr[i][2])
            else:
                lst_demands.append(self.arr[i][2])
        return lst_demands


def max_flow(graph: DW_Graph, s, t) -> int:
    """
        The following returns the maximum flow for a Flow Network (i.e. Connected, Directed Graph)

        Time Complexity: O(EF) where E is the number of edges and F is the maximum flow.
    """
    flow = 0
    visited = [False] * graph.number_vertices
    # Call DFS. It'll return the "Residual Capacity" along the augmenting path (p) it found (i.e. The min. edge capacity in p):
    augment = DFS(s, t, float("inf"), visited, graph)
    # Keep incremenetung the "Residual Capacity" found in subsequent augmenting path's as 
    # That's the amount our total flow is increasing by through going into that augmenting path:
    while augment > 0:
        flow += augment
        visited = [False] * graph.number_vertices
        augment = DFS(s, t, float("inf"), visited, graph)
    return flow

def DFS(u, t, bottleneck, visited: list, graph: DW_Graph):
    """
        The following function is an implementation of DFS whereby
        we go through and try to get to the the target (t) vertex from the source vertex along
        some path. If we can achieve this, find the minimum capacity edge along that path (p)
        through the min(bottleneck, residual) and return it.

        Complexity: O(E) where E is the number of edges
    """
    # Once we've hit the target node (t), we have an augmenting path.
    # This path will contain in variable "bottleneck" the "Residual Capacity"
    # (i.e. min capacity edge in this path). Return it.
    if u == t:
        return bottleneck
    visited[u] = True
    # Get all adjacent edges, including reverse edges.
    adjacent_edges = graph.get_adjacent_edges(u)
    for edge in adjacent_edges:
        # For each edge, try see if you can get a residual > 0. 
        # If we keep getting residuals > 0 until a target then we'll return
        # a bottleneck that is != 0 and increase total flow.
        v = edge.connected_to
        capacity = edge.capacity
        flow = edge.flow
        residual = capacity - flow
        if residual > 0 and not visited[v]:
            # Call DFS and return the augment (i.e. "Resdiual Capacity").
            # This is done through always keeping the minimum capacity edge
            # value through min(bottleneck, residual) (i.e. min of all previous edge residuals and current edge residual).
            augment = DFS(v, t, min(bottleneck, residual), visited, graph)
            if augment > 0:
                # If augment > 0, then we know that we should update the "flow" values of all our edges.
                # This just does that, for each edge, it'll update the flow value.
                edge.flow += augment
                # Example: If you push 2 units of flow through forward edge, now you can withdraw two units of flow (i.e.
                # 2 units of flow will go into the reverse edge where flow is negative in reverse edge).
                edge.reverse_edge.flow -= augment
                return augment
    # If no augmenting path, return 0.
    return 0