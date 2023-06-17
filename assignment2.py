"""
    Please Note for Part 1 Documentation:   The terms "vertex" and "data centre" are used interchangably.
                                            The terms "edge" and "communication channel" are also used interchangably.
                                            The terms "flow" and "throughput" are also used interchangably.
"""
class CircularQueue():
    """
        Circular Queue Implementation
    """

    def __init__(self, capacity: int) -> None:
        """
            Method Description: This function Initializes an array to represent the Queue.
            
            Input: 
                capacity: Maximum number of elements that can be stored in Queue.
            
            Time Complexity: O(N) where N is the size of the array.
            Aux Space Complexity: O(N) where N is the size of the array.
        """
        self.array = [None] * capacity
        self.front = 0
        self.end = 0
        self.length = 0

    def add(self, element) -> None:
        """
            Method Description: Pushes an element to the front of the queue.

            Input:
                element: An item of any type to be pushed into the Queue

            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        # Storing an element at the rear of the Queue, wrapping around the array if needed:
        self.array[self.end] = element
        self.end = (self.end + 1) % len(self.array)
        self.length += 1

    def serve(self):
        """
            Method Description: Retrieves and removes the element at the front of the queue.

            Output: The item that was previously at the front of the Queue

            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        self.length -= 1
        item = self.array[self.front]
        self.front = (self.front + 1) % len(self.array)
        return item
    
    def __len__(self) -> int:
        """
            Method Description: Returns the length of the Queue.

            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        return self.length

class Edge:
    """
        A weighted, directed edge in a Flow Network which defines a relationship TO v from some other vertex u,
        with a flow and a capacity.
    """

    def __init__(self, u: int, v: int, c: int) -> None:
        """
            Method Description: This function Initializes an Edge Object.

            Inputs: 
                u: The vertex where the Edge is an outgoing Edge.
                v: The vertex where the Edge is an incoming Edge.
                c: The capacity of the Edge, defining the max amount of flow that can pass through this edge.
            
            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        self.connected_from = u
        self.connected_to = v
        self.flow = 0              # Initially, the flow of an edge is always 0.
        self.capacity = c

    def add_reference_reverse_edge(self, reverse_edge) -> None:
        """
            Method Description: For a given edge, this function adds an attribute reverse_edge.

            Input:
                reverse_edge: An Edge object which references the reverse edge for a given edge,
                representing the back edge in the Residual Network.

            Example: If there is an edge from u to v, it would have an attribute reverse_edge that references the edge v to u

            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        self.reverse_edge = reverse_edge

class Flow_Network:
    """
        Implementation of a Flow Network that has been modified to account for the 
        parameters of the assignment by adding relevant methods.
    """

    def __init__(self, number_vertices: int) -> None:
        """
            Method Description: This method takes an adjacency list implementation to representing the flow network.

            The following understanding is needed for the following class methods and understanding how the flow network is constructed:
            This method takes in a given number_vertices and for each vertex, initializes 3 copy vertices.
            Given vertex v, one copy vertex represents the "Input" vertex where all edges coming into v is directed to this "Input" vertex.
            One copy represents the "Normal" vertex which has the sum of all Inputs flowing into v and sum of all outputs flowing out of v.
            The final copy represents the "Output" vertex which has all the edges flowing out of v to other vertices.

            Example: If we wanted to initialize vertices in [0, 1], we'd have to create 2*3 = 6 vertices like [0,1,2,3,4,5]
            The index of the "Input" vertex of vertex 0 is just 0*3=0. The index of the "Normal" vertex of vertex 0 is just (0*3)+1=1. 
            The index of the "Output" vertex of vertex 0 is just (0*3)+2=2.

            In general, the index to access a vertices:
            "Input Vertex" in the adjacency list: (vertex_no * 3).
            "Normal Vertex" in the adjacency list: (vertex_no * 3) + 1
            "Output Vertex" in the adjacency list: (vertex_no * 3) + 2

            Input:
                number_vertices: The number of vertices in the graph if there was one vertex for each vertex.

            Attributes:
                arr: A list of tuples where each tuple holds at index 0 the vertex ID and at index 1, the list of adjacent edges.
                number_vertices: The number of vertices in the graph if we used 3 vertices to represent a single vertex v.

            Time Complexity: O(3n) = O(n), where n is the number of vertices to initialize
            Aux Space Complexity: O(3n) = O(n), where n is the number of vertices to initialize
        """
        self.arr = [(v, []) for v in range(number_vertices * 3)]
        self.arr.append((len(self.arr), []))   # Appending the super-target vertex.
        self.number_vertices = (number_vertices * 3) + 1

    def add_communication_channel(self, u: int, v: int, c: int, super_channel: bool = False) -> None:
        """
            Method Description: This method adds a communication channel between vertices u and v, with a
            capacity of c. It subsequently initializes the reverse communication channel to go from vertex v to u,
            representing the backwards edge in a Residual Flow Network.

            Inputs:
                u: The vertex where the Edge is an outgoing Edge.
                v: The vertex where the Edge is an incoming Edge.
                c: The capacity of the Edge, defining the max amount of flow that can pass through this edge.
                super_channel: If True, we're adding an edge from our target u to the super target v.
                If False, we're simply adding an edge from vertex u's Output vertex to vertex v's Input vertex.

            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        # The following gets the index of the "Input" vertex for v
        # and "Output" vertex for u to be connected with an edge.
        if super_channel is False:
            u = (u*3) + 2
            v = (v*3)
        else:
            u = (u*3) + 1    # Initializing the "Normal" vertex for u.
        # Adding forward and reverse edges between u and v to adjacency list:
        forward_edge = Edge(u, v, c)
        backwards_edge = Edge(v, u, 0)
        forward_edge.add_reference_reverse_edge(backwards_edge)
        backwards_edge.add_reference_reverse_edge(forward_edge)
        self.arr[u][1].append(forward_edge)
        self.arr[v][1].append(backwards_edge) 

    def add_maxInp_constraint(self, u: int, maxInp: int) -> None:
        """
            Method Description: For a given vertex u, this method sets the Maximum sum of Inputs constraint by 
            adding an edge between the "Input" vertex for u and "Normal" vertex for u. This edge constrains
            the maximum sum of inputs to be the value of the argument maxInp because only maxInp can flow
            from the "Input" to "Normal" vertex.

            In essence, the purpose of the "Input" vertex is to take in all the flow and have an edge directed
            to the "Normal" vertex that constrains the sum of all these input flows.

            Inputs:
                u: The vertex ID to set the maximum input constraint for.
                maxInp: The maximum sum of inputs a vertex u can take from incoming edges/communication channels.
            
            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        # creating edge between "Input" and "Normal" vertex u.
        forward_edge = Edge(u*3, (u*3)+1, maxInp) 
        backwards_edge = Edge((u*3)+1, u*3, 0)
        forward_edge.add_reference_reverse_edge(backwards_edge)
        backwards_edge.add_reference_reverse_edge(forward_edge)
        # Adding edge to Flow Network.
        self.arr[u*3][1].append(forward_edge)
        self.arr[(u*3)+1][1].append(backwards_edge)

    def add_maxOut_constraint(self, u: int, maxOut: int) -> None:
        """
            Method Description: For a given vertex u, this method sets the Maximum sum of Outputs constraint by 
            adding an edge between the "Normal" vertex for u and "Output" vertex for u. This edge constrains
            the maximum sum of output to be the value of the argument maxOut because only maxOut can flow
            from the "Normal" vertex to the "Output" vertex, which gives the "Output" vertex only maxOut flow to 
            push through its outgoing edges/communication channels.

            Inputs:
                u: The vertex ID to set the maximum output constraint for.
                maxOut: The maximum amount of flow a vertex can push out from vertex u.
            
            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        forward_edge = Edge((u*3)+1, (u*3)+2, maxOut)
        backwards_edge = Edge((u*3)+2, (u*3)+1, 0)
        forward_edge.add_reference_reverse_edge(backwards_edge)
        backwards_edge.add_reference_reverse_edge(forward_edge)
        self.arr[(u*3)+1][1].append(forward_edge)
        self.arr[(u*3)+2][1].append(backwards_edge)

    def get_adjacent_edges(self, u: int) -> list:
        """
            Method Description: Retrieves all adjacent edges from a vertex u by returning a list of adjacent edges.

            Input:
                u: The vertex ID of a vertex.

            Output: A list containing all adjacent edges to vertex u (i.e. outgoing edges from vertex u).

            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        return self.arr[u][1]


def maxThroughput(connections: list[tuple], maxIn: list[int], maxOut: list[int], origin: int, targets: list[int]) -> int:
    """
        Function Description: This function returns the maximum possible data throughput/flow from the origin to the data
        centres specified in the targets input array, given certain constraints (maxIn, maxOut and capacties of connections).

        Approach Description: To construct the Flow Network, for each vertex v, add 2 additional vertices v_in and v_out where 
        v_in has a communication channel flowing into v, representing the associated maxIn (maximum amount of data that can flow
        into data centre v). v should then have a communication channel flowing into v_out, representing the associated maxOut
        for vertex v (maximum amount of data that can flow out of data centre v). We also add a super-target vertex which
        connects to all target vertices (t1, t2, etc...) so that we can set a single target which takes the sum of all the flows
        coming into the target vertices t1, t2, etc...

        Once the Flow Network has been constructed, run the max_flow algorithm, which as seen through its function description,
        will perform Breadth-First Search (BFS) on the Flow Network to find the SHORTEST Augmenting Path from the origin
        to the super-target. It'll keep performing this BFS on the Flow Network until no more augmenting paths can be found, meaning
        no more additional data can be pushed through communication channels from origin to the super-target.
        In each iteration of this BFS on the Flow Network, a counter keeps track of the bottleneck (i.e. the minimum capacity edge
        along the augmenting path) which represents how much we increase the flow of data by along the augmenting path. The sum of 
        all bottlenecks throughout all the BFS calls is the maximum Throughput/flow of data for a given Flow Network.

        Inputs:
            connections: A list of tuples containing details of the connections between two data centres and the capacity of flow.
            maxIn: A list of integers representing the max flow of data that can possibly go into a vertex.
            maxOut: A list of integers representing the max flow of data that can possibly go out of a vertex.
            origin: The source data centre where the data initially flows from.
            targets: A list of all target data centers where flow will end up.

        Output: An integer representing the maximum total flow of data (throughput) that is capable of being pushed from the source data centre s 
        to the target data centre t.

        NOTE: For the sake of simplicty, I omitted the multiples in Big-O notation in the following complexity explanations.
        For Example: I might've wrote O(|D|) instead of O(3|D|) as O(3|D|) = O(|D|).

        Time Complexity Explanation: Where |D| is the number of data centres and |C| is the number of communication channels.
            Initializing the Flow Network costs O(|D|) worst-case time, as per its documentation.
            Adding connections to the Flow Network costs O(|C|) worst-case time as there's |C| communication channels and each channel costs O(1) time to add.
            Adding the maxInp and MaxOut costs O(2|D|) worst-case time as there's |D| data centres and each data centre maxIn or maxOut constraint
            costs O(1) time to add.
            Adding the connections to the super-target costs O(|D|) worst-case time as there's a most, |D|-1 data centres to connect to the super-target, because we don't consider the origin.
            Finally, calling the max_flow function costs O(|D|x|C|^2) worst-cast, as per its documentation.
            The worst-case time complexity: O(|D| + |C| + 2|D| + |D|) + O(|D|x|C|^2) = O(|D|x|C|^2)
        

        Aux Space Complexity Explanation: Where |D| is the number of data centres and |C| is the number of communication channels.
            Initializing the Flow network costs O(|D|) worst-case aux space complexity, as per its documentation.
            Adding connections to the Flow Network costs O(|C|) worst-case aux space complexity as there's at most |C| connections.
            Adding the maxInp and maxOut constraints costs O(|D|) worst-case aux space complexity as there's at most |D| data centres.
            Adding the connections to the super-target costs O(|D|) worst-case aux space complexity as there's at most |D| data centres.
            Finally, calling the max_flow function costs O(|D| + |C|) worst-case aux space complexity, as per the function documentation.
            The worst-case aux space complexity: O(|D| + |C| + |D| + |D| + |D| + |C|) = O(|D| + |C|)

        Time Complexity: O(|D|x|C|^2) where C is the number of Communication Channels and D is the number of Data Centres.
        Aux Space Complexity: O(|D| + |C|) where C is the number of communication Channels and D is the number of Data centres.
    """
    # Construction of Flow Network:
    number_data_centres = len(maxIn)
    flow_network = Flow_Network(number_data_centres)
    # Initializing connections within graph:
    for connection in connections:
        flow_network.add_communication_channel(connection[0], connection[1], connection[2])
    # Initializing constraints of maxIn and maxOut for all vertices except the super-target:
    for i in range(len(maxIn)):
        flow_network.add_maxInp_constraint(i, maxIn[i])
    for i in range(len(maxOut)):
        flow_network.add_maxOut_constraint(i, maxOut[i])
    # Add connections from the "Normal" target vertices to a newly created super-target.
    super_target_index = len(flow_network.arr) - 1     # super-target is stored at end of flow_network adjacency list.
    for target_index in targets:
        flow_network.add_communication_channel(target_index, super_target_index, maxIn[target_index], super_channel = True)
    return max_flow((origin * 3) + 1, (super_target_index), flow_network)   # Calling max_flow from the origin's "Normal" vertex to the super-target.


def max_flow(s: int, t: int, graph: Flow_Network) -> int:
    """
        Function Description: This function is an implementation of Ford-Fulkerson using BFS (Breadth-First Search) whereby we 
        call the Ford_Fulk_BFS to find the SHORTEST AUGMENTING PATH given the current state of the Flow Network from s to t and
        push as much data through this path as possible. We keep calling Ford_Fulk_BFS until no more augmenting paths
        can be found from s to t, returning the sum of the flow of data.

        Inputs:
            s: The source data centre where the data initially flows from.
            t: The super-target data centre which represents the sum of all incoming flow to the target(s) vertices.
            graph: A Flow Network representing the data centres and their associated communication channels.

        Output: An integer representing the maximum total flow of data (throughput) that is capable of being pushed from the source data centre s 
        to the target data centre t.

        Time Complexity Explanation: Where |D| is the number of data centres and |C| is the number of communication channels.
            A single call of the function Ford_Fulk_BFS costs O(|C|) worst-case time, as seen through its documentation.
            The number of function calls to Ford_Fulk_BFS depends on the number of augmenting paths we can find in the Flow Network. Since BFS (Breadth-First Search) always finds the SHORTEST AUGMENTING PATH, there's less communication channels along this path, reducing the likelihood of smaller bottlenecks.
            This means we end up finding at most |C|*|D| augmenting paths and hence, iterating through the while loop |C|*|D| times.
            O(|C|*|C|*|D|) = O(|D|*|C|^2)

        Aux Space Complexity Explanation: Where |C| is the number of communication channels.
            This algorithm calls Ford_Fulk_BFS which uses up O(|C|) worst-case aux space.

        Time Complexity: O(|D|*|C|^2) where C is the number of Communication Channels and D is the number of Data Centres.
        Aux Space Complexity: O(|D| + |C|) where C is the number of Communication Channels and D is the number of data centres.
    """
    max_flow = 0
    bottleneck = Ford_Fulk_BFS(s, t, graph)
    while bottleneck > 0:
        max_flow += bottleneck
        bottleneck = Ford_Fulk_BFS(s, t, graph)
    return max_flow

def Ford_Fulk_BFS(s: int, t: int, graph: Flow_Network) -> int:
    """
        Function Description: This function find the SHORTEST path in a Flow Network from the source s to the target t (augmenting path).
        We then call the Augment function to push as much additional data flow through the augmenting path found as possible and return the bottleneck pushed through.

        Inputs:
            s: The source data centre where the data initially flows from.
            t: The super-target data centre which represents the sum of all incoming flow to the target(s) vertices.
            graph: A Flow Network representing the data centres and their associated communication channels.

        Output: An integer representing the bottleneck (i.e. min capacity of communication channels along the augmenting path found)
        which is the amount of additional data flow pushed through the augmenting path in a single run of this function.

        Time Complexity Explanation: Where |D| is the number of data centres and |C| is the number of communication channels.
            The initializations cost O(3|D|) = O(|D|) worst-case because for each data centre, we initialized 3 data centres a "Input", "Normal" and "Output" data centre.
            The while loop costs O(3|D| + 3|C|) = O(|D| + |C|) worst-case because we serve at most (3|D|) data centres and in the worst-case,
            must visit all (3|C|) communication channels to find an augmenting path from s to t.
            Finally, the Augment Function call costs O(|C|) worst-case as specified in its documentation.
            O(|D| + |D| + |C| + |C|) = O(|D| + |C|) = O(|C|) as D = O(C)

        Aux Space Complexity Explanation: Where |D| is the number of data centres and |C| is the number of communication channels.
            The initializations cost O(|D|) worst-case because we intialize a few arrays, each of which contains at most |D| data centres.
            Furthermore, the CircularQueue holds a max capacity of |D|.
            The while loop costs O(1) aux space as it uses no additional data structures.
            Finally, the Augment Function call costs O(|C|) worst-case as specified in its documentation
            O(|C|) + O(|D|) = O(|D| + |C|)

        Time Complexity: O(|C|) where C is the number of Communication Channels.
        Aux Space Complexity: O(|D| + |C|) where C is the number of communication Channels and D is the number of data centres.
    """
    # Initializations:
    reached_target = False     # Did we reach the target node?
    predecessor_array = [None] * graph.number_vertices
    visited_bit_lst = [0] * graph.number_vertices     # bit list of visited data centres.
    visited_bit_lst[s] = 1
    queue = CircularQueue(capacity=graph.number_vertices)
    queue.add(s)
    while len(queue) != 0:
        u = queue.serve()
        # For each data centre u, get it's adjacent communication channels:
        adjacent_channels = graph.get_adjacent_edges(u)
        for channel in adjacent_channels:
            # Calculating the residual (i.e. amount of data that can still be pushed through
            # a particular communication channel)
            v = channel.connected_to
            capacity = channel.capacity
            flow = channel.flow
            residual = capacity - flow
            # If an adjacent channel has not be visited yet and can still have data pushed through it,
            # visit the data centre that is on the end of this adjacent channel.
            if visited_bit_lst[v] != 1 and residual > 0:
                visited_bit_lst[v] = 1
                queue.add(v)
                predecessor_array[v] = u
                # Check: Once we've hit the super-target data centre, the shortest augmenting path is found.
                if v == t:
                    reached_target = True
        if reached_target:  # Can stop iteration once shortest augmenting path found.
            break
    bottleneck = Augment(s, t, predecessor_array, reached_target, graph)
    return bottleneck

def Augment(s: int, t: int, predecessor_array: list, reached_target: bool, graph: Flow_Network) -> int:
    """
        Function Description: This function reconstructs the edges/communication channels along the augmenting path taken to get from source (s) 
        to target (t) in a single run of the Ford-Fulkerson (BFS) algorithm by using the predecessor_array.
        It then will find the bottleneck (i.e. min capacity of communication channels along the augmenting path) and adjust
        the flow of data through the communication channels by increasing data flow by the bottleneck along forward edges in the augmenting path,
        and decreasing data flow by the bottleneck the corresponding reverse edges.

        Inputs:
            s: The source data centre where the data initially flows from.
            t: The super-target data centre which represents the sum of all incoming flow to the target(s) vertices.
            predecessor_array: A list describing the order of vertices/data centres traversed to get from s to t.
            reached_target: A boolean where True means the BFS found an augmenting path from s to t and False means BFS failed to find such a path.
            graph: A Flow Network representing the data centres and their associated communication channels.

        Output: An integer representing the bottleneck (i.e. min capacity of communication channels along the augmenting path found)
        which is the amount of additional data flow pushed through the augmenting path in a single iteration of the Ford_Fulk_BFS.

        Time Complexity Explanation: Where |D| is the number of data centres and |C| is the number of communication channels.
            Step 1 in the algorithm costs O(|D|) worstcase as we iterate through at most |D| data centres to go from s to t.
            Step 2 costs O(|C|) as in the worst-case, we iterate through all |C| communication channels (edges) to find the ones
            which exist along the augmenting path.
            Step 3 costs O(|C|) worst-case as we're simply iterating through the edges along the augmenting path.
            Step 4 costs O(|C|) worst-case. Same explanation with step 3.
            O(|D|) + O(3|C|) = O(|C|)

        Aux Space Complexity Explanation: Where |D| is the number of data centres and |C| is the number of communication channels.
            Step 1 in the algorithm costs O(|D|) worstcase as we're creating an array of data centres.
            Step 2 in the algorithm costs O(|E|) worstcase as we're creating an array of communication channels.
            Steps 3 & 4 cost O(1) aux space.
            O(|D|) + O(|C|) = O(|C|)

        Time Complexity: O(|C|) where C is the number of Communication Channels
        Aux Space Complexity: O(|C|) where C is the number of Communication Channels
    """
    bottleneck = 0
    # If we found an augmenting path to the target, bottleneck > 0 by performing the following steps:
    if reached_target:
        # Step 1: Using predecessor array, reconstruct the vertices traversed to get from s to t:
        vertices_path = [t]
        while s != t:
            predecessor_vertex = predecessor_array[t]
            vertices_path.append(predecessor_vertex)
            t = predecessor_vertex
        vertices_path.reverse()

        # Step 2: Using those vertices, find edges traversed to get from one vertex to the next in the augmenting path.
        # Use the fact that vertex v and (v+1) have an edge between them where 0 <= v < (number vertices in path) - 1,
        # Check the adjacent edges for each vertex in the augmenting path to find the appropriate edges.
        edges_path = []
        for i in range(len(vertices_path) - 1):
            v = vertices_path[i]
            lst_adjacent_edges = graph.get_adjacent_edges(v)
            for edge in lst_adjacent_edges:
                if edge.connected_to == vertices_path[i+1]:
                    edges_path.append(edge)

        # Step 3: Iterate through the augmenting path taken to find the bottleneck.
        bottleneck = float("inf")
        for edge in edges_path:
            if (edge.capacity - edge.flow) < bottleneck:
                bottleneck = (edge.capacity - edge.flow)
        
        # Step 4: Update data flow values for each edge along the augmenting path, in addition to their reverse edges.
        for edge in edges_path:
            edge.flow += bottleneck
            edge.reverse_edge.flow -= bottleneck
    return bottleneck



# NOTE: PART 2:

"""
    Please note that TrieNode/Node/Word are all interchangable terminology used to describe a word in a sentence as words are stored in a TrieNode.
"""

def index_of_character(character: str) -> int:
    """
        Function Description: This function takes a character of the English alphabet or $ as input and returns the ord value
        or 26 for $, representing the index of where to store the link to that character in a TrieNode array.

        Input:
            character: A string of a single character representing one of the cats words.

        Output: An integer representing the position to store this single character
        in an array that contains links to all of the potential next characters in the sentence.

        Time Complexity: O(1) as we're just comparing a single character string.
        Aux Space Complexity: O(1)
    """
    if character == "$":
        return 26
    return ord(character) - 97

class TrieNode:
    """
        Trie Node Implementation, representing a Node within the Trie data structure.
    """
    def __init__(self, character: str) -> None:
        """
            Method Description: This method initializes a TrieNode for storing information about a specific character in a string
            (i.e. cat word in a sentence).

            Input:
                character: A string of a single character representing one of the cats words.
            
            Attributes:
                children_array: An array that provides at most 27 links to "children" (i.e. possible next words in the sentence) where
                "a" is stored at index 0, "b" at index 1, ..., "z" at index 25, "$" at index 26.
                character: A string of a single character representing one of the cats words.
                most_occurrences: An integer representing the length of the LONGEST sentence that can be traversed to/searched for
                by going through the TrieNode.
                optimal_child: A link to another TrieNode which represents at any word in the Trie, the "child" word in the children_array that is the most frequently occurring in any sentence and, in the case of multiple sentences having the same occurrence,
                is the lexicographically smaller child word.

            Time Complexity and Aux Space Complexity Explanations:
            As we're only setting attributes and initializing and array of fixed size (27), the complexity is O(1).

            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        self.children_array = [None] * 27
        self.character = character
        self.most_occurrences = 1
        self.optimal_child = None

    def get_children_nodes(self, node) -> list:
        """
            Method Description: This method takes a TrieNode as input and returns a list of all its "children" nodes meaning
            all the possible words that could go next, after the word stored in the input TrieNode.

            Input:
                node: A TrieNode, containing information about a particular word within the Trie data structure

            Output: A list containing the "children" nodes of a given node/word in the Trie.
            
            Time Complexity and Aux Space Complexity Explanations:
            As we're performing a fixed number of iterations and appending in each iteration, complexity is O(1).

            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        lst_children = []
        for i in range(27):
            if node.children_array[i] is not None:
                lst_children.append(node.children_array[i])
        return lst_children

class CatsTrie:
    """
        CatsTrie Implementation, which contains all the sentences, with each word
        in the sentence stored as a character of the alphabet in a TrieNode.
    """
    def __init__(self, sentences: list[str]) -> None:
        """
            Function Description: This function initializes the Trie data structure through firstly, inserting all the
            sentence in the input sentences list into the Trie data structure. After a single sentence is inserted, the second step in this function is to traverse the path of the sentence that was just inserted to update the most_occurrences attribute of each TrieNode in the path IF the sentence is not a newly inserted sentence. The final step is that the function will traverse the path of the sentence again to update the optimal_child attribute of each TrieNode in the path. The above 3 steps which are performed on EACH SENTENCE being inserted guarantees the information held inside each TrieNode through its attributes is always accurate.

            Inputs:
                sentences: A list of strings containing sentences of characters where each sentence represents a cat sentence.

            Time Complexity Explanation: Where |N| is the number of sentence in sentences and |M| is the number of characters in the longest sentence:
                This function will iterate over each sentence in sentences through the first for loop, costing O(|N|) best/worstcase time.
                Nested within this looping over the sentences is:
                    A join method, which joins the original sentence with a "$" character at the end, costing O(|M|) worstcase time.
                    An insertion of a single sentence of worst-case length |M|, costing O(|M|) worstcase time.
                    An iteration through this sentence to update most_occurrences attribute, costing O(|M|) worstcase time.
                    A final iteration through this sentence to update optimal_child attribute, costing O(|M|) worstcase time.

                The overall worst-case time complexity is: O(|N| * 4|M|) = O(|N| * |M|)

            Aux Space Complexity Explanation: Where |N| is the number of sentence in sentences and |M| is the number of characters in the longest sentence:
                This function uses a join method to add a "$" to the end of our sentence string. As the method must create and store a new string
                of worst-case size |M+1|, the aux space complexity is O(|M|). No further additional data structures are used.

            Time Complexity: O(|N| * |M|) where |N| is the number of sentence in sentences and |M| is the number of characters in the longest sentence
            Aux Space Complexity: O(|M|) where |M| is the number of characters in the longest sentence
        
        """
        self.root = TrieNode("#")      # Initalizing root.
        for sentence in sentences:
            sentence = "".join([sentence, "$"])    # O(M)
            # Step 1: Performing insertion of Nodes. O(M)
            (freq_string_occurrences, new_sentence) = self.insertion(sentence, self.root)

            # Step 2: If inserting a sentence already existing in the Trie, for each character/TrieNode
            # along the path of the sentence, update its most_occurrences attribute to always hold the
            # number of occurrences of the most frequently occurring sentence reachable from that character/TrieNode:
            # O(M) as iterating through a single sentence.
            if not new_sentence:
                node = self.root
                for character in sentence:
                    if freq_string_occurrences > node.most_occurrences:
                        node.most_occurrences = freq_string_occurrences
                    character_index = index_of_character(character)
                    node = node.children_array[character_index]
            
            # Step 3: Traverse same path, updating the link of the optimal_child.
            # O(M) as iterating through a single sentence.
            node = self.root
            for character in sentence:
                lst_children = node.get_children_nodes(node)
                # Iterating through all children is O(27) = O(1) as 27 potential children at any given TrieNode.
                for child_node in lst_children:
                    # Conditionals:
                    # 1. If the most_occurrences of a child == most_occurrences of its parent
                    # and it is lexicographically the smallest child, set it to be the optimal child.
                    # 2. Otherwise, if most_occurrences of a child node == most_occurrences of its parent and
                    # it has a most_occurrences greater than that of the current optimal child, we know that the
                    # child_node is infact the optimal child.
                    if (child_node.most_occurrences == node.most_occurrences) and \
                    (ord(child_node.character) < ord(node.optimal_child.character)):
                        node.optimal_child = child_node
                    elif (child_node.most_occurrences == node.most_occurrences) and \
                    (child_node.most_occurrences > node.optimal_child.most_occurrences):
                        node.optimal_child = child_node
                # Move onto the next node in the path.
                character_index = index_of_character(character)
                node = node.children_array[character_index]

    def insertion(self, sentence: str, root: TrieNode) -> tuple[int, bool]:
        """
            Function Description: This function inserts all the characters (i.e. words) within a sentence into the Trie by
            effectively searching the Trie for the sentence. If it finds the entire sentence through finding the relevant
            sequence of words that make up that sentence, no insertions of new words are required. Otherwise, we're inserting a 
            new sentence and so search the Trie for that sentence and the moment you can't search anymore (i.e. found a prefix
            for that sentence), start inserting new characters/words into the Trie from the last character of that prefix.
    
            Inputs:
                sentence: A string containing a sentence of characters that represents a cat sentence.
                root: A TrieNode representing the starting node where its "children" TrieNodes all represent the first word in a given sentence. 
                For example, it's children might be "a" and "c" because the sentences we have are ["aye", "cat"].

            Output: A tuple containing the current number of occurrences of a given sentence AND a boolean, indicating whether we
                    had to insert a new sentence (True) or found an identical sentence already in the Trie (False).

            Time Complexity Explanation: Where |M| is the number of characters in the longest sentence.
                This function iterates through all characters in a sentence. Since the length of the longest sentence is |M|,
                it performs |M| iterations at most. Furthermore, in each iteration of |M|, there's only constant time operations being performed.

            Time Complexity: O(|M|) where M is the number of characters in the longest sentence.
            Aux Space Complexity: O(1) as additional data structures of variable length are used.
        """
        node = root
        inserted_new_sentence = False

        # Iterate through all characters (i.e. words) in a sentence. O(M)
        for character in sentence:
            character_index = index_of_character(character)
            # If the TrieNode we're currently looking at does not have a "child" linked with the current character,
            # This means we should insert a new character, and provide a link to it from the TrieNode.
            if node.children_array[character_index] is None:
                tmp_node = TrieNode(character)
                node.children_array[character_index] = tmp_node
                node.optimal_child = tmp_node            # Set its child to optimal child
                node = tmp_node                          # Move down to newly created node.
                if node.character == "$":
                    inserted_new_sentence = True
            # If the TrieNode already has the character we're looking at as a child, simply move down to that child.
            else:
                node = node.children_array[character_index]
                if node.character == "$":
                    # If traversed through entire sentence without inserting, sentence already existed in Trie before. +1 to number occurences of 
                    # this sentence
                    node.most_occurrences += 1
        return (node.most_occurrences, inserted_new_sentence)

    def searching(self, prompt: str) -> TrieNode:
        """
            Function Description: This function searches a given prompt (i.e. prefix) and will return the TrieNode containing information about
            the last character in the prompt if it can find the prompt in the Trie. Otherwise, it can't find the prompt in the Trie.
    
            Inputs:
                prompt: A string representing the prefix of a sentence to be searched for in the Trie.

            Output: Either the TrieNode containing information about the last character in the prompt or a None object.

            Time Complexity Explanation: Where |X| is the length of the prompt.
                This function iterates through all the characters in the prompt of length |X|, performing constant time
                operations in each iteration.

            Time Complexity: O(|X|) where |X| is the length of the prompt.
            Aux Space Complexity: O(1) as additional data structures of variable length are used.
        """
        # Edge Case: If provided prompt is empty string, return the root TrieNode.
        if prompt == "":
            return self.root
        # Otherwise: Return an Inner/Leaf TrieNode.
        node = self.root
        node_to_return = None
        final_character = prompt[-1]     # The target character to be searched for.
        
        # Iterate through prompt, returning None if at any stage if we can't find the next character. O(|X|)
        for character in prompt:
            character_index = index_of_character(character)
            if node.children_array[character_index] is None:
                return None
            # Otherwise, the character exists, so move onto next.
            else:
                node = node.children_array[character_index]
                if node.character == final_character:   # O(1) when comparing single characters.
                    node_to_return = node
        return node_to_return

    def autoComplete(self, prompt: str) -> str:
        """
            Function Description: This function returns an output string that represents the completed sentence of a given prompt
            where this prompt denotes how the sentence should start (i.e. prefix of the sentence) via a Trie data structure.

            Approach Description: Since the prompt denotes how the sentence should start (i.e. prefix of the sentence), we can 
            attempt to search for this prompt and see if we can find a sequence of words in the Trie that match up to the input prompt.
            If the search can't find an identical sequence in the Trie, we know that none of the sentences can be used to finish the prompt
            as the prompt does not appear as a prefix for any of the sentences. We can stop the algorithm there and return.
            However, if the search does find an identical sequence to the prompt inside the Trie, the search will return the TrieNode that represents 
            the last character in the prompt. From this last character, perform the following:
                1. Use the attribute optimal_child which links to a "child" TrieNode to always get the TrieNode that should be traversed to next from a given TrieNode. This optimal_child attribute, as described in the __init__ method of the class TrieNode will always link to the child of a given TrieNode that allows the function to traverse to the sentence which has the greatest frequency from a given TrieNode and in the case of tie-breakers, returns the lexicographically smaller "child" TrieNode.
                2. Keep traversing using the optimal_child link until the algorithm reaches a "$" TrieNode, denoting the end of the sentence as we're told "$" won't be in the 26-letter alphabet. For
                each TrieNode traversed, store its character attribute so that we know the next words in the completed sentence.
                3. Join the original prompt string with all the characters that were traversed through in steps 1-2 above. Return this joined string, representing the completed sentence.

            Input:
                prompt: A string representing the prefix of a sentence to be searched for in the Trie.

            Output: A string representing the appropriate completed sentence from the prompt.

            Time Complexity Explanation: Where |X| is the length of the prompt and |Y| is the length of the most frequent sentence in sentences that begins with the prompt:
                There are two cases. Firstly, if the prompt is not found in the Trie we simply stop the function after using the Trie search method to search for the prompt. This costs O(|X|) worstcase where X is the length of the prompt.
                Secondly, if the prompt is found in the Trie, a while loop is used to continuously get to the next node in the most frequent 
                sentence that begins with the prompt. This while loop costs O(|Y|) worstcase. Once the while loop is over,
                a join method is used, which costs O(|X| + |Y|) as |X| + |Y| is the worstcase length of string to be joined.

                Therefore, the time-complexity is O(|X|) if such a prompt (input) doesn't exist in the Trie and O(2|X| + 2|Y|) = O(|X| + |Y|) otherwise.

            Aux Space Complexity Explanation: Where |X| is the length of the prompt and |Y| is the length of the most frequent sentence in sentences that begins with the prompt:
                This function uses a join method to join a string of length |X| and a string of length |Y| in their worstcase. As the method must create and store a new string of worst-case size |X| + |Y|, the aux space complexity is O(|X| + |Y|). No further additional data structures are used.

            Time Complexity: O(X) if the prompt (input) doesn't exist in the Trie and O(|X| + |Y|) if the prompt (input) exists in the Trie, where
            |X| is the length of the prompt and |Y| is the length of the most frequent sentence that begins with the prompt.
            Aux Space Complexity: O(|X| + |Y|) where |X| is the length of the prompt and |Y| is the length of the most frequent sentence in sentences that begins with the prompt.
        """
        last_node = self.searching(prompt)      # Searching prompt costs O(X) 
        # If couldn't find prompt in Trie:
        if last_node is None:
            return None

        # Otherwise, continue searching the Trie for the appropriate way to complete the sentence using the optimal_node
        # until you reach the end of the sentence. This costs O(Y).
        completed_sentence_lst = [prompt]
        node = last_node.optimal_child 
        while node.character != "$":
            completed_sentence_lst.append(node.character)
            node = node.optimal_child                        # Traversing to next optimal_child.

        # Use a join to return and get the completed sentence: O(X + Y)
        completed_sentence = "".join(completed_sentence_lst)
        return completed_sentence