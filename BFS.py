from graph_class import Graph
import queue

def breadth_first_search(Graph_Obj: Graph, s: int):
    """
        The following is an implementation of Breadth-First Search

        Complexity: O(V + E)
    """
    # Initialize bit_lst of size length of graph
    visited_bit_lst = [0] * Graph_Obj.no_vertices
    visited_bit_lst[s] = 1
    Q = queue.Queue()
    Q.append(s)
    while len(Q) != 0:
        u = Q.serve()
        # For each vertex v adjacent to u (i.e. contained in Graph.arr[u]):
        # Accesses the list of adjacent vertices
        for i in range(len(Graph_Obj.arr[u][1])):
            Edge = Graph_Obj.arr[u][1][i]  # Adjacent Edge
            v = Edge.connected_to
            # If Adjacent edge hasn't been visited yet
            if visited_bit_lst[v] != 1:
                visited_bit_lst[v] = 1
                Q.append(v)

    return visited_bit_lst