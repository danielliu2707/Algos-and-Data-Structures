# Ford-Fulkerson Implementation with BFS (Edmonds Karp Algorithm):

class CircularQueue():
    """
    Implementation of a Circular Queue
    """

    def __init__(self, max_capacity: int) -> None:
        """
            Initializes an array with a maximum capacity of elements to represent the Queue.
            
            Input: 
                max_capacity (int): Maximum number of elements that can be stored in Queue.

            Attributes:
                length (int): Number of items currently in the Queue
                front (int): The index of the current start of the Queue
                rear (int): The index of the current end of the Queue
                array (list): The array representation of the Queue
        """
        self.array = [None] * max_capacity
        self.front = 0
        self.rear = 0
        self.length = 0

    def push(self, element) -> None:
        """
            Pushes an element to the end of the queue.

            Precondition: Checks to ensure we're not pushing to a full queue

            Input:
                element: The item to be pushed into the Queue

            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        if self.length == len(self.array):
            raise Exception("Can't push to a full queue")
        self.array[self.rear] = element
        self.length += 1
        self.rear = (self.rear + 1) % len(self.array)

    def serve(self):
        """
            Retrieves and removes the element at the front of the queue

            Precondition: Checks to ensure we're not serving from an empty queue

            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        if self.length == 0:
            raise Exception("Can't serve from an empty queue")
        self.length -= 1
        item = self.array[self.front]
        self.front = (self.front+1) % len(self.array)
        return item
    
    def __len__(self) -> int:
        """
            Returns the length of the Queue
        """
        return self.length

class Edge:
    """
        A weighted, directed edge in a Flow Network which defines a relationship TO v from some other vertex u,
        with a flow and a capacity.
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

# NOTE: Modify to no longer require demand. Rename to something more relevant like Flow_Network.
class DW_Graph:
    """
        A directed, weighted graph class modified to account for flow, capacity and demand
    """

    def __init__(self, number_vertices):
        """
            Initializes graph with n vertices.
            Vertex IDs go from 0 to n-1 (inclusive).
        """
        # NOTE: Need to have one array like [0, 1, 2, 3, 4, 5, 6]. The index of an item is just 0//3=0 1//3=0, 2//3=0, 3//3=1, 4//3=1, 5//3=1, 6//3=2.
        # this means that IDs 0, 1 and 2 still map to vertex 0.
        self.arr = [(v, []) for v in range(number_vertices * 3)]   # Initializing an Input, Normal and Output Vertex.
        self.number_vertices = number_vertices

    def add_communication_channel(self, u, v, f, c):
        """
            Adds weighted edge between (u, v) of weight w.
            This means u is directed to v (i.e. stored in A[u] is Edge(v)).
            Initialize the corresponding reverse edge.

            We're adding a direct communicatiion channel connection from an output vertex (u) to input vertex (v).
        """
        forward_edge = Edge(u, v, f, c)
        backwards_edge = Edge(v, u, 0, 0)   # Initialize flow = 0, capacity = 0. Flow of backwards edges (i.e. flow to be redirected) is negative
        forward_edge.add_reference_reverse_edge(backwards_edge)
        backwards_edge.add_reference_reverse_edge(forward_edge)
        self.arr[u][1].append(forward_edge)
        self.arr[v][1].append(backwards_edge)   # Adding backwards edge to list of edges

    def add_maxInp_constraint(self, u, maxInp):
        """
            For a vertex u, add the max sum of Inputs constraint from its Input Vertex to Normal Vertex
        """
        forward_edge = Edge(u*3, (u*3)+1, 0, maxInp)
        backwards_edge = Edge((u*3)+1, u, 0, 0)
        forward_edge.add_reference_reverse_edge(backwards_edge)
        backwards_edge.add_reference_reverse_edge(forward_edge)
        self.arr[u*3][1].append(forward_edge)
        self.arr[(u*3)+1][1].append(backwards_edge)

    def add_maxOut_constraint(self, u, maxOut):
        """
            For a vertex u, add the max sum of Outputs constraint from its Normal Vertex to Output Vertex
        """
        forward_edge = Edge((u*3)+1, (u*3)+2, 0, maxOut)
        backwards_edge = Edge((u*3)+2, (u*3)+1, 0, 0)
        forward_edge.add_reference_reverse_edge(backwards_edge)
        backwards_edge.add_reference_reverse_edge(forward_edge)
        self.arr[(u*3)+1][1].append(forward_edge)
        self.arr[(u*3)+2][1].append(backwards_edge)

    def get_adjacent_edges(self, u) -> list:
        """
            Retrieves all adjacent edges from a vertex u by returning a list of edges
            which u is connected to (i.e. Output version of Vertex u).
            Modified to not only include outgoing edges for u, but also
            Incoming edges for u, but with the reverse flow. 
            O(1)
        """
        return self.output_vertices_arr[(u//3)+2][1]

def max_flow(graph: DW_Graph, s, t) -> int:
    """
        The following returns the maximum flow for a Flow Network (i.e. Connected, Directed Graph)

        Time Complexity: O(EF) where E is the number of edges and F is the maximum flow.
    """
    flow = 0
    # Call DFS. It'll return the "Residual Capacity" along the augmenting path (p) it found (i.e. The min. edge capacity in p):
    augment = BFS(s, t, graph)
    # Keep incremenetung the "Residual Capacity" found in subsequent augmenting path's as 
    # That's the amount our total flow is increasing by through going into that augmenting path:
    while augment > 0:
        flow += augment
        augment = BFS(s, t, graph)
    return flow

def BFS(s, t, graph: DW_Graph):
    """
        The following function is an implementation of BFS whereby
        we go through and try to get to the target (t) vertex from the source vertex along
        the shortest path which is achieved through BFS. If we can achieve this,
        find the minimum capacity edge along that path (p) and augment the path with that minimum capacity.
        We find the path through a reconstruction.

        Algorithm:
        1. Perform BFS from source to target, adding an edge only if residual (i.e. remaining capacity > 0).
        2. Stop as soon as any of the edges hits the target.
        3. Reconstruct this augmenting path.
        4. Find the minimum capacity of any edge along this augmenting path.
        5. Augment the path with that minimum capacity, updating both forward and reverse edges.
        6. Perform steps 1-5 until no more augmenting paths from source to target can be found.

        Inputs:
            s: The source vertex.
            t: The target vertex.
            graph: A directed weighted graph representing a flow network.
        
        Output: The minimum capacity of any edge along the augmenting path. Represents the increase in flow along this augmenting path.

        Complexity: O(|V| + 2|E|) = O(|E|) where E is the number of edges
    """

    reached_target = False     # Did we reach the target node?
    predecessor_array = [None] * graph.number_vertices
    visited_bit_lst = [0] * graph.number_vertices
    visited_bit_lst[s] = 1
    queue = CircularQueue(max_capacity=graph.number_vertices)
    queue.push(s)
    # O(|V| + |E|) = O(|E|)
    while len(queue) != 0:
        u = queue.serve()
        # For each vertex u, get it's adjacent edges:
        adjacent_edges = graph.get_adjacent_edges(u)
        for edge in adjacent_edges:
            v = edge.connected_to
            capacity = edge.capacity
            flow = edge.flow
            residual = capacity - flow
            # If Adjacent edge hasn't been visited yet and has a residual > 0
            if visited_bit_lst[v] != 1 and residual > 0:
                visited_bit_lst[v] = 1
                queue.push(v)
                predecessor_array[v] = u
                # Checking if we reached the target: Can stop BFS
                if v == t:
                    reached_target = True
        # If we reached the target, can stop performing BFS:
        if reached_target:
            break
    augment_capacity = Augment(s, t, predecessor_array, reached_target, graph)   #O(|E|)
    return augment_capacity

def Augment(s, t, predecessor_array: list, reached_target: bool, graph: DW_Graph) -> int:
    """
        This function augments the path, updating flow values of both the forward
        and reverse edges along the augmenting path. It returns the capacity of the
        min capacity edge along the augmenting path.

        Complexity: O(|E|) where E is the number of edges
    """
    min_capacity = 0
    # If we found an augmenting path to the target, min_capacity edge will be greater than 0.
    if reached_target:
        # Step 1: Get all edges a list of all vertices traversed in the path using predecessor_array:
        # O(|V|) as you iterate through at most all |V| vertices in the predecessor_array
        vertices_path = [t]
        while s != t:
            predecessor_vertex = predecessor_array[t]
            vertices_path.append(predecessor_vertex)
            t = predecessor_vertex
        vertices_path.reverse()

        # Step 2: Get a list of corresponding edges traversed in the path:
        # len(vertices_path) - 2 to skip over the target.
        # For each vertex in the path, check over all of its adjacent edges to see if there's an edge
        # which we traversed through in the augmenting path, and append that edge. O(|E|) as you'll traverse through at most  
        # all |E| edges in an augmenting path.
        edges_path = []
        for i in range(len(vertices_path) - 1):
            v = vertices_path[i]
            lst_adjacent_edges = graph.get_adjacent_edges(v)
            # Check ove
            for edge in lst_adjacent_edges:
                if edge.connected_to == vertices_path[i+1]:
                    edges_path.append(edge)

        # Step 3: Iterate through list of edges to find the minimum capacity amongst those edges: O(|E|)
        min_capacity = float("inf")
        for edge in edges_path:
            if edge.capacity < min_capacity:
                min_capacity = edge.capacity

        # Step 4: Augment along the path with that minimum capacity: O(|E|)
        for edge in edges_path:
            edge.flow += min_capacity
            edge.reverse_edge.flow -= min_capacity
    return min_capacity