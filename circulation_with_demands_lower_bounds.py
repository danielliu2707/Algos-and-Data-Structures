# Implementation of circulation with demands and lower bound problem:

# Feasibility problem which identifies if a graph satisfies the Capacity constraint and Demand constraint.

# Circulation with demands and lower bounds:
from ford_fulkerson_dfs import max_flow

class Edge:
    """
        A weighted, directed edge which defines a relationship TO v from some other vertex with flow f, lower bound l and capacity c.
    """

    def __init__(self, u, v, f, l, c):
        self.connected_from = u
        self.connected_to = v
        self.flow = f
        self.lower_bound = l
        self.capacity = c

    def add_reference_reverse_edge(self, reverse_edge):
        """
            Connects an edge to another edge. This method is used when connecting an edge to its
            corresponding reverse edge.
        """
        self.reverse_edge = reverse_edge


class DW_Graph:
    """
        A directed, weighted graph class modified to account for flow, capacity, lower bounds and demand.
        It is therefore a representation of a flow network.
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

    def add_weighted_edge(self, u, v, f, l, c):
        """
            Adds weighted edge between (u, v) of weight w.
            This means u is directed to v (i.e. stored in A[u] is Edge(v)).
            Initialize the corresponding reverse edge
        """
        forward_edge = Edge(u, v, f, l, c)
        backwards_edge = Edge(v, u, 0, -1 * float("inf"), 0)
        forward_edge.add_reference_reverse_edge(backwards_edge)
        backwards_edge.add_reference_reverse_edge(forward_edge)
        self.arr[u][1].append(forward_edge)
        self.arr[v][1].append(backwards_edge)   # Adding backwards edge to list of edges
        self.forward_edges[u][1].append(forward_edge)    # Appending to array of only forward edges to get original flow network

    def get_adjacent_edges(self, u) -> list:
        """
            Retrieves all adjacent edges from a vertex u by returning a list of edges.
            Modified to not only include outgoing edges for u, but also
            Incoming edges for u, but with the reverse flow.
        """
        return self.arr[u][1]

    def get_all_edges(self, forward_edges = False) -> list:
        """
            Retrieves all edges within a graph.

            Arguments:
                forward_edges (bool): Specifies if we only want forward edges or not:

            Time Complexity: O(|E|).
        """
        # Getting all edges in graph:
        if not forward_edges:
            lst_edges = []
            # Get all adjacent edges for each vertex. Go through this list of adjacent edges and append to overall lst_edges:
            for i in range(len(self.arr)):
                adjacent_edges = self.get_adjacent_edges(self.arr[i][0])
                for edge in adjacent_edges:
                    lst_edges.append(edge)
        # Getting all forward edges in a graph:
        else:
            lst_edges = []
            # Get all adjacent edges for each vertex. Go through this list of adjacent edges and append to overall lst_edges:
            for i in range(len(self.forward_edges)):
                for edge in (self.forward_edges[i][1]):
                    lst_edges.append(edge)
        return lst_edges

    def get_all_demands(self, only_positive = False, only_negative = False) -> list:
        """
            Returns a list of all demand values, allowing optional arguments to get only positive or negative demands.
            If both optinal arguments are False, gets all demands.
        """
        lst_demands = []
        for i in range(len(self.demands)):
            if only_positive:
                if self.demands[i] > 0:
                    lst_demands.append(self.demands[i])
            elif only_negative:
                if self.demands[i] < 0:
                    lst_demands.append(self.demands[i])
            else:
                lst_demands.append(self.demands[i])
        return lst_demands

def circulations_with_demand(graph: DW_Graph) -> bool:
    """
        The following checks whether the given graph with demands can obtain a feasible solution.
        Returns True if there is a feasible solution. Returns False if no feasible solution.
    """
    # Step 1: Check pre-condition. If sum of demands != 0, no feasible solution
    demands = graph.get_all_demands()
    if sum(demands) != 0:
        return False
    # Step 2: Add super-source s which has an outgoing edge to all vertices with negative demands.
    # Iterate through all vertices, setting an edge from source s to it if there's a negative demand.
    number_vertices = graph.number_vertices
    graph.arr.append(("s", []))
    graph.forward_edges.append(("s", []))
    for i in range(number_vertices):
        # If negative demand, set edge from s with the positive demand:
        demand = graph.demands[i]
        if demand < 0:
            graph.add_weighted_edge(number_vertices, i, 0, 0, -1 * demand)   # Note: At index number_vertices sits source s
    # Step 3: Do the same but with incoming edges going into t
    graph.arr.append(("t", []))
    graph.forward_edges.append(("s", []))
    for i in range(number_vertices):
        # If positive demand, set edge going to t:
        demand = graph.demands[i]
        if demand > 0:
            graph.add_weighted_edge(i, number_vertices + 1, 0, 0, demand)  # Note: At index number_vertices + 1 sits target t
    # Step 4: Run Ford-Fulkerson, returning max_flow
    graph.number_vertices += 2    # Increment number_vertices as we added "s" and "t"
    flow = max_flow(graph, number_vertices, number_vertices + 1)
    # Step 5: If flow equal to sum of positive demands, then return True as a feasible solution is possible:
    positive_demands = graph.get_all_demands(only_positive = True)
    if flow == sum(positive_demands):
        return True
    return False

def circulation_with_lower_bounds(graph: DW_Graph):
    """
        The following checks whether a graph is a feasible solution to the
        circulation with demands and lower bounds problem. To be so, it'd have to 
        satisfy the capacity constraint and demand constraint.
    """
    # Modify graph to remove dependence on lower bounds:

    # Iterate through edges, updating capacity of that edge (capacity = capacity - lower bound),
    # and the demand of vertices it's connected to. (+1) to demand for vertex it's coming from.
    # (-1) to demand for vertex it's going to:
    lst_forward_edges = graph.get_all_edges(forward_edges = True)   # Getting all forward edges
    for edge in lst_forward_edges:
        # Updating capacity of that edge:
        edge.capacity -= edge.lower_bound
        # Updating demand of vertices connected to that edge:
        vertex_u = edge.connected_from
        vertex_v = edge.connected_to
        graph.demands[vertex_u] += 1
        graph.demands[vertex_v] -= 1
    if circulations_with_demand(graph) is True:
        # Return Final Solution (Adding back lower bounds to graph):
        # Edges: Add lower bound back to capacity and to flow
        # Demands: Do opposite of above for demands
        # Flow: Requires modification of both ford_fulkerson and circulation with demands to
                # return the graph itself with its flows.
        return True
    else:
        return False