import heapq

# Part 1: Should I give a ride?


class Edge:
    """
        Class Description: This class defines a relationship from some vertex u to some other vertex v, with weigh w.
    """

    def __init__(self, u, v, w):
        self.connected_from = u
        self.connected_to = v
        self.weight = w


class DW_Graph:
    """
        Class Description: This class represents a Directed, Weighted Graph using an Adjacency List.
        The adjacency list is a list containing tuples of (vertex, lst_edges) where each lst_edges contains all the edges that particular
        vertex is adjacent to (i.e. points to) in the graph.

    """

    def __init__(self, number_vertices: int) -> None:
        """
            Method Description: This method initializes a graph with a certain number of vertices whereby
            those vertices will have vertex ID's ranging from 0 to number_vertices - 1. Each vertex has an
            associated list within its tuple, which will refer to all of its adjacent edges.

            Time Complexity: O(n), where n is the number of vertices to initialize
            Aux Space Complexity: O(n), where n is the number of vertices to initialize
        """
        self.arr = [(v, []) for v in range(number_vertices)]

    def add_weighted_edge(self, u: int, v: int, w: int) -> None:
        """
            Method Description: This method adds a weighted edge between vertices u and v.

            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        self.arr[u][1].append(Edge(u, v, w))

    def get_adjacent_edges(self, u: int) -> list:
        """
            Method Description: This method returns a list of all edges adjacent to the input vertex.

            Input: A vertex ID.
            Output: A list of all adjacent edges to the input vertex ID.

            Time Complexity: O(1)
            Aux Space Complexity: O(1)
        """
        return self.arr[u][1]


def get_number_locations(roads: list) -> int:
    """
        Function Description: This function returns the number of locations (i.e. Vertices) that are present within
        the graph. It does so by iterating through all the roads (i.e. edges), finding the highest numbered Location
        that was mentioned in the roads (both to and from a location) and returning this highest numbered location.

        Example: If I had locations represented by {0, 1, ..., 10}, the highest numbered location is 10.

        Given R is the number of roads:

        The loop costs O(R) best/worst case in time since we're just linear searching through all |R| roads.

        Input: A list of all roads
        Output: The highest numbered Location.

        Time Complexity: O(R) where R is the number of roads.
        Aux Space Complexity: O(1)
    """
    largest_location_id = 0

    # "linear search" through roads input, checking whether a road contains a location which has
    # a greater id than the current max id
    for i in range(len(roads)):
        starting_location = roads[i][0]
        ending_location = roads[i][1]
        if starting_location > largest_location_id:
            largest_location_id = starting_location
        if ending_location > largest_location_id:
            largest_location_id = ending_location
    largest_location_id += 1
    return largest_location_id


def build_graph(roads: list, is_carpool: bool, should_reverse: bool) -> DW_Graph:
    """
    Function Description: This function builds a Directed Weighted Graph, using either Carpool, or Non-carpool travel times
    and, allowing us, should we so choose, to reverse the direction of the roads as specified in the roads input. It builds
    the Directed Weighted Graph by initializing a Directed Weighted Graph, and iterating through all the roads, and adding
    a Directed, Weighted Edge between two Locations within the Graph for each road.

    Given R is the number of roads and L is the number of locations.

    Initializing the DW_Graph costs O(L) as we initialize |L| vertices in the graph.

    The loop costs O(R) best/worst case in time since we're just iterating through all |R| roads.

    Inputs:
        - roads: A list of roads which join two locations and have travel times.
        - is_carpool: A boolean telling us whether to build a graph with weighted edges based on carpool or noncarpool travel times
        - should_reverse: A boolean telling us whether to reverse the direction of all roads specified in the roads input.

    Output: A Directed Weighted Graph object.

    Time Complexity: O(R) where R is the number of roads.
    Aux Space Complexity: O(L) where L is the number of locations.
    """
    number_of_locations = get_number_locations(roads)
    graph = DW_Graph(number_of_locations)

    # Iterating through all roads, and adding weighted edges.
    for i in range(len(roads)):
        starting_location = roads[i][0]
        ending_location = roads[i][1]
        # Checking whether to use carpool or non-carpool weighted edges
        if is_carpool:
            distance = roads[i][3]
        else:
            distance = roads[i][2]
        # Checking whether insert edges normally or in the reverse direction what was specified in roads input
        if should_reverse:
            graph.add_weighted_edge(
                ending_location, starting_location, distance)
        else:  # if not reversing
            graph.add_weighted_edge(
                starting_location, ending_location, distance)
    return graph


def dijkstra(graph: DW_Graph, source: int, number_locations: int) -> tuple:
    """
    Function Description: This function implements Dijkstra's algorithm whereby we hold a priority queue to continuously
    ensure the Location with the smallest distance to the source is as the top of the queue. Initially, our priority queue
    has the source location, and whilst there's still locations with their distances in our queue stored as a tuple, we do the following:
    Get the location with the smallest distance from the source in the priority queue, access all of its adjacent Edges.
    For each adjancent edge, "relax" on all their distances and update distances in the queue.

    Given R is the number of roads and L is the number of locations.

    Initializing the distance and predecessor array is O(R).

    The while loop costs O(L) as we keep looping until all locations have had their distances finalised.
        - Getting the min item from priority queue costs O(log L) as we must take the last item in the heap, swap it to the top of the heap
        and make it "fall down" into its correct position which at worst is O(log L) as there's a height of log L for the binary heap.

    The for loop costs O(R) as in its worstcase, we test all the roads present in our graph.
        - Updating a locations distance in the priority queue costs O(log L).

    Inputs:
        - graph: A directed Weighted Graph.
        - source: An integer to represent the starting/source location.
        - number_locations: An integer to represent the number of locations.

    Output: A tuple containing the finalised (Distance Array, Predecessor Array)

    Time Complexity: O(R log(L)) where R is the number of roads and L is the number of locations.
    This is because the complexity of O(R log(L) + L log(L)) = O(R log(L)) since R dominantes L because in a dense graph, R = O(L^2).

    Aux Space Complexity: O(R + L) where R is the number of roads and L is the number of locations.
    This is because we use distance and predecessor arrays taking up O(L) space and the
    function get_adjacent_edges returns a list of adjacent edges (i.e. Roads), costing O(R) space.
    """
    distance = [float("inf")] * number_locations
    predecessor = [None] * number_locations
    distance[source] = 0

    priority_queue = []
    # Initially, push (distance, location) onto heap
    heapq.heappush(priority_queue, (0, source))

    # The while loop iterates through all Locations, retrieving the minimum distance from source
    # location in each iteration
    while len(priority_queue) != 0:
        # Gets tuple: (distance, location)
        min_location = heapq.heappop(priority_queue)
        adjacent_edges = graph.get_adjacent_edges(      # Retrieves list of adjacent edges to location
            min_location[1])

        # The for loop checks all adjacent edges, and "relaxes" them by setting a shorter distance, if
        # possible for that location in the priority queue
        for edge in adjacent_edges:
            location_u = edge.connected_from
            location_v = edge.connected_to
            weight = edge.weight
            if distance[location_v] > distance[location_u] + weight:
                distance[location_v] = distance[location_u] + weight
                predecessor[location_v] = location_u
                # Updated distance set in priority queue
                heapq.heappush(priority_queue, (weight,  location_v))
    return (distance, predecessor)


def reconstruct(start: int, end: int, predecessor: list, is_reversed: bool) -> list:
    """
    Function Description: This function reconstructs the path taken to get from some starting location to some ending location.
    It does so by going to the end location, and finding its predecessor location. Then for this location you find the predecessor again.
    This keeps going until you reach the starting location. Whilst in this loop, you append the locations you've past over to an array,
    to then be reversed to get the correct order.

    Given R is the number of roads and L is the number of locations.

    The while loop costs O(L) as at most, there'll be |L| items to backtrack through

    The reverse operation at the end costs O(L) as it's just reversing a list of at most |L| items.

    Inputs:
        - start: The starting source location.
        - end: The ending target location.
        - predecessor: The array produced from Dijkstra's giving us the predecessor of each location.
        - is_reversed: A boolean indicating whether the Directed Weighted Graph had its edges in "reverse" order from what was specified in roads input

    Output: A list containing the sequence of locations visited to get from the start to the end location.

    Time Complexity: O(L), where L is the number of locations
    Aux Space Complexity: O(L), where L is the number of locations
    """
    path = []
    backtrack_from = end
    # Iterate through all the predecessor locations from the end, until you get to the start.
    while backtrack_from != start:
        # If using reverse edged graph, don't append the passenger node (as it'll be repeated twice)
        if backtrack_from == end and is_reversed:
            backtrack_from = predecessor[backtrack_from]
        else:
            path.append(backtrack_from)
            backtrack_from = predecessor[backtrack_from]
    path.append(start)
    # If not a reversed edge graph, reverse the order to get right order for the path
    if not is_reversed:
        path.reverse()
    return path


def optimalRoute(start: int, end: int, passengers: list, roads: list) -> list:
    """
    Function Description: This function returns the optimal route taken to go from the start location to end location
    where optimal means to minimise the travel time. It returns an output array in the correct order in which we
    must've passed through the locations to get this minimal travel time from start to end.

    Approach Description:
    There is two potential cases. Firstly, if there are no "passengers" (i.e. empty passengers array),
    we know that we'll never get onto the "carpool" (i.e. potentially faster) lanes and therefore, we may call Dijkstra on a
    Normal Directed Weighted Graph with its nodes representing passengers, edges representing roads and weight representing travel time.
    From calling Dijkstra, we get the shortest path from start to end using the "noncarpool" lane travel times as weights
    and then we can reconstruct the path it took by looking into the predecessor array.

    The second case is if there are "passengers" (i.e. a non-empty passengers array). We know that potentially,
    we'll go past a location where we pick up a passenger which allows us to travel on the "carpool" (i.e. potentially faster) lanes.
    So we firstly call Dijkstra on a Normal Directed Weighted Graph, noting down the optimal travel time to all
    the locations with passengers at it. Then, we call Dijkstra again, but now on a Directed Weighted Graph with
    its edges (i.e. roads) pointing in the opposite direction to what was initially specified in the "roads" input array.
    This allows us to call Dijkstra from the "end" location as our source, giving us the optimal distances to locations from the end
    location. We should note down the optimal travel time from the "end" location to the "passenger" locations, using the faster "carpool" lanes because after you pick up a passenger, you can travel on the carpool lanes.

    If we then add up the optimal travel time from the source location to a passenger location
    (i.e. what we noted down with first Dijkstra call)) with the optimal travel time from the ending location to that same passenger location
    (i.e. what we noted down with second Dijkstra call on graph with edges in reverse direction),
    this gives us the optimal time whilst going through that particular passenger location. Doing this calculation for all passenger locations,
    and taking the minimum of such travel times gives us the overall optimal travel time. Finally, using this knowledge, we may reconstruct the optimal path we took going through that particular passenger location which gives us the minimum travel time.

    There is an edge case I deal with at the end. If there are passenger locations BUT avoiding them would give the optimal route, 
    meaning we only take non-carpool lanes, then take only non-carpool lanes.

    Given R is the number of roads and L is the number of locations.

    Explanation (Time Complexity):
    The function build_graph costs O(R) in time, as mentioned in its function description.

    The function get_number_locations costs O(R) in time, as mentioned in its function description.

    The function dijkstra costs O(R log(L)) in time, as mentioned in its function description.

    The function reconstruct costs O(L) in time, as mentioned in its function description.

    Prior to the first for loop, the above 4 functions are called individually, meaning the current time complexity is O(R log(L)).

    The first for loop in this main function, we iterate through all |P| passengers, performing O(1) addition and indexing each iteration.
    The first for loop is O(|P|) which means time complexity so far is O(R log(L) + P) but since R = O(P) as
    the |P| can never exceed |R| and where R = O(P), we say O(R log(L)) is the dominant term. Thus, time complexity is still O(R log(L))
    after the first for loop.

    The second for loop, is very much the same as the first for loop, with a same complexity. Therefore, overall complexity is
    still O(R log(L)).

    Finally, we call reconstruct twice, which costs O(2L) time. Our overall complexity = O(R log(L) + 2L) = O(R log(L)).


    Explanation (Space Complexity):
    The function build_graph costs O(L) in aux space, as mentioned in its function description.

    The function get_number_locations costs O(1) in aux space, as mentioned in its function description.

    The function dijkstra costs O(R + L) in aux space, as mentioned in its function description.

    The function reconstruct costs O(L) in aux space, as mentioned in its function description.

    Prior to the first for loop, the above 4 functions are called individually, meaning the current aux space
    complexity is O(R + L)

    The first for loop in this main function creates an additional array of size |P| which, in its worst-case is equal to |L|.
    Thus, P = O(L). This means that the aux space complexity is O(R + 2L) = O(R + L).

    The second for loop modifies values within an existing array. It is inplace, taking O(1) aux space.

    Finally, we call reconstruct twice which costs O(2L) in aux space, meaning
    our overall complexity is = O(R + 3L) = O(R + L).

    Inputs:
        - start: The starting source location.
        - end: The ending target location.
        - passengers: A list containing the location of all passengers
        - roads: A list of all roads

    Output: A list containing the sequence of locations visited to get from the start to the end location in the shortest travel time.

    Time Complexity: O(R log(L)) where R is the number of roads and L is the number of locations.
    Aux Space Complexity: O(R + L) where R is the number of roads and L is the number of locations.
    """
    # Build Directed Weighted Graph with non-reversed edges which use non-carpool travel times.
    non_carpool_graph = build_graph(
        roads, is_carpool=False, should_reverse=False)

    # Build Directed Weighted Graph with reversed edges which use carpool travel times.
    carpool_graph = build_graph(roads, is_carpool=True, should_reverse=True)

    number_locations = get_number_locations(roads)

    distance_non_carpool, pred_non_carpool = dijkstra(
        non_carpool_graph, start, number_locations)

    # If pedestrian array is empty, simply reconstruct graph from end to source (Case 1):
    passenger_len = len(passengers)
    if passenger_len == 0:
        optimal_path = reconstruct(start, end, pred_non_carpool, is_reversed = False)
    # Otherwise, perform the following (Case 2):
    else:
        # Call Dijkstra from the end value as the source, with my edges reversed.
        distance_carpool, pred_carpool = dijkstra(
            carpool_graph, end, number_locations)

        # For each pedestrian location, do the following summation:
        # (min distance from source location to pedestrian location) + (min distance from end location to pedestrian location).
        # Store the results in an array. These results represent the distances to go from the source to the end location
        # Whilst traversing through a location where we pick up a passenger.
        sum_distances_array = []
        for i in range(passenger_len):
            passenger = passengers[i]
            # Get distance from source to the passenger using non-carpool lanes
            distance_1 = distance_non_carpool[passenger]
            # Get distance from end to passenger using carpool lanes
            distance_2 = distance_carpool[passenger]
            # Take sum and append
            sum_distances_array.append(distance_1 + distance_2)

        # For each distance calculated in above, find the min optimal distance and the associated
        # passenger we went through to get that min.
        min_distance = float('inf')
        min_passenger = None
        for i in range(len(sum_distances_array)):
            if sum_distances_array[i] < min_distance:
                min_distance = sum_distances_array[i]
                min_passenger = passengers[i]

        # Edge Case: When avoiding passenger locations gives the optimal route:

        # If avoiding passenger locations gives a shorter distance than going through
        # any passenger location:
        if distance_non_carpool[end] < min_distance:
            optimal_path = reconstruct(
                start, end, pred_non_carpool, is_reversed=False)
        else:
            # Otherwise, going through a passenger location gives the optimal route:

            # Get the path from the start location to this min_passenger location
            optimal_path_start = reconstruct(
                start, min_passenger, pred_non_carpool, is_reversed=False)

            # Get the path from the end location to this min_passenger location
            optimal_path_end = reconstruct(
                end, min_passenger, pred_carpool, is_reversed=True)

            # Join start of path with end of path to get overall path
            optimal_path = optimal_path_start + optimal_path_end

    return optimal_path


# Part 2: Repurposing Underused Workspace


def reconstruct_solution(memo_array: list, number_rows: int, number_columns: int, occupancy_probability: list) -> list:
    """
        Function Description: This function reconstructs my solution from bottom to top by finding the minimum total occupancy
        in the last row of the memo_array matrix at location (i, j) which represents the section in the last row to be removed.
        Then my function goes to the row above, either to the location (i-1, j-1), (i-1, j) or (i-1, j+1) as those are the
        possible locations I could've came from to get to location (i, j). It decides which of those 3 locations to go to
        based on how to get the optimal solution in memo_array at location (i, j). The function keeps doing this until it gets
        to the top row, finally returning a list of all the locations it went through to get the overall minimum total occupancy.
        This employs the "backtracking" method of reconstructing my solution.

        Given N is the number of rows and M is the number of columns:

        Time Complexity:
        Getting the minimum total occupancy in the final row costs O(M) time as there's M values in a row.

        Going back up the previous row from the bottom of our memo_array matrix to the top costs O(N) time as there's N rows.

        Aux Space Complexity:
        Building the array of locations to be removed costs O(N) space as there's N sections to be removed.

        Inputs:
            memo_array: Finalised memo array containing all the min. distances to that location within the NxM seating arrangements.
            number_rows: Number of rows in the memo array
            number_columns: Number of columns in the memo array
            occupancy_probability: A list of lists (i.e. matrix) containing N interior lists representing N rows
            and where the interior lists are all of length M, representing M columns. The value at
            occupancy_probability[i][j] where i is the row and j is the column represents an integer between
            0 and 100.


        Output: A list containing 2 items. The first item is an integer which represents the total minimum
        occupancy for the selected N sections to be removed. The second item is a list of N tuples where each
        tuple is in the form (i, j), with i representing the row number and j representing the column number
        of a section to be removed in order to get our total minimum occupancy value.

        Time Complexity: O(N + M) = O(N) as N > M (as per the specifications). N is the number of rows.
        Aux Space Complexity: O(N) where N is the number or rows.
    """

    # The min total occupancy is the min of the last row in memo_array.
    # The optimal_location is the Optimal solution's (row, column) location.
    # O(M) as this loop is taking the minimum of M values in last row.
    minimum_total_occupancy = memo_array[number_rows-1][0]
    optimal_location = (number_rows-1, 0)
    for j in range(1, number_columns):
        if memo_array[number_rows-1][j] < minimum_total_occupancy:
            minimum_total_occupancy = memo_array[number_rows-1][j]
            # Get the location of where in the final row the minimum_total_occupancy occurs
            optimal_location = (number_rows-1, j)

    # In memo_array, try go back up either to the location directly above (i-1, j), up and left (i-1, j-1) or up and right (i-1, j+1)
    # depending on whichever location I came from whilst building the memo_array to get to location (i, j).
    # O(N) as there's N rows to go back up from the bottom of the memo_array to the top of memo_array. O(1) operations in each iteration.
    selections_location = []
    for _ in range(number_rows):
        selections_location.append(optimal_location)

        # Get the current optimal row and column location within the tuple
        optimal_row = optimal_location[0]
        optimal_column = optimal_location[1]

        # Get previous rows optimal total occupancy value which I used to get to location (i, j)
        optimal_occupancy_prev_row = memo_array[optimal_row][optimal_column] - \
            occupancy_probability[optimal_row][optimal_column]

        # Check if we came from location (i-1, j) to get to location (i, j):
        if memo_array[optimal_row-1][optimal_column] == optimal_occupancy_prev_row:
            optimal_location = (optimal_row-1, optimal_column)

        # Check if we came from location (i-1, j-1) to get to location (i, j):
        elif (optimal_column-1 >= 0) and (memo_array[optimal_row-1][optimal_column-1] == optimal_occupancy_prev_row):
            optimal_location = (optimal_row-1, optimal_column-1)
        # Otherwise, we came from location (i-1, j+1) to get to location (i, j):
        else:
            optimal_location = (optimal_row-1, optimal_column+1)

    selections_location.reverse()

    return [minimum_total_occupancy, selections_location]


def select_sections(occupancy_probability: list) -> list:
    """
        Function Description: This function builds a memo matrix of size N x M from the input occupancy_probability
        matrix of size N x M through the use of dynamic programming. The function will then find the location
        where the the minimum total occupancy exists in the final row of the N x M memo matrix and reconstruct
        the path of locations taken to get to the location where this minimum total occupancy exists.

        Approach Description: We initialize our memo_array of size N x M, where an entry at location (i, j) within
        the memo_array (i.e. memo_array[i][j]) represents the optimal sum of occupancy probabilities to get to that (i, j) location
        whilst taking a path from a section in the top row in our original occupancy_probability matrix to that (i, j) location.

        I will initialize the first row in the memo_array (i.e. memo_array[0][0], memo_array[0][1], ... memo_array[0][j]) to be their corresponding values in the occupancy_probabilities matrix. This is because the sum of occupancy probabilities to get to location (0, 0), (0, 1), ... (0, j)
        is simply just the occupancy probability as it's the first row.

        Values in subsequent rows in the memo_array is calculated by looking at the row above in the memo_array (i.e. at row i-1), and taking the
        minimum of the values at location (i-1, j-1), (i-1, j) or (i-1, j+1) because they represent the minimum sum of occupancy probabilities up until the point of (i-1, j-1), (i-1, j) or (i-1, j+1) AND they are locations are in the same of adjacent column as location (i, j), but just in the row above. Adding this minimum value to the occupancy_probability at location (i, j) gives the minimum sum of occupancy probabilities to get to location (i, j). By applying this method to each row, I fill each row in my memo_array from left-to-right until I have a memo_array of size N x M filled.

        After building the memo_array as such, I can reconstruct my solution by calling the reconstruct_solution function.
        A description of this is provided in the function description of reconstruct_function.

        Given N is the number of rows and M is the number of columns:

        Time Complexity:
        Initializing the memo_array costs O(NM) time as there is N x M entries.

        Initializing first row in memo_array is O(M) time as there's M entries per row.

        Building the memo_array costs O(NM) time as there is N x M entries to fill in, with each
        entry costing O(1) time. Each entry costs O(1) time because we just to perform some comparisons
        on integers, perform some arithmetic and take the minimum either 2 or 3 values, which is constant for each entry.

        Thus total Time Complexity: O(NM + M) = O(NM)

        Aux Space Complexity:
        Building a memo_array of size N x M costs O(NM) aux space.

        Input:
            occupancy_probability: A list of lists (i.e. matrix) containing N interior lists representing N rows
            and where the interior lists are all of length M, representing M columns. The value at
            occupancy_probability[i][j] where i is the row and j is the column represents an integer between
            0 and 100.

        Output: A list containing 2 items. The first item is an integer which represents the total minimum
        occupancy for the selected N sections to be removed. The second item is a list of N tuples where each
        tuple is in the form (i, j), with i representing the row number and j representing the column number
        of a section to be removed in order to get our total minimum occupancy value.

        Time Complexity: O(NM) where N is the number of rows and M is the number of columns.
        Aux Space Complexity: O(NM) where N is the number of rows and M is the number of columns.
    """
    # Get number of rows and columns in occupancy probability matrix.
    number_rows = len(occupancy_probability)
    number_columns = len(occupancy_probability[0])

    # Initialize memo_array to appropriate size: O(NM) time/space as N x M entries.
    memo_array = [([None] * number_columns) for i in range(number_rows)]

    # BaseCase: Initialize first row in memo_array to be original values
    # in occupancy_probability matrix's first row: O(M) time as M entries in first row.
    for j in range(number_columns):
        memo_array[0][j] = occupancy_probability[0][j]

    # Building of Memo: Build from left to right, row by row.
    # For each entry in Memo array, store the optimal sum of occupancy probabilities
    # to get to that (i, j) position within the N x M matrix.
    # O(NM) time as there is N x M entries to fill in Memo array and O(1) work in each iteration:
    for i in range(1, number_rows):
        for j in range(number_columns):
            # If memo_array[i][j] has a value in memo_array (directly above it), (to its left and up) and (to its to right and up).
            if (j-1 >= 0) and (j+1 <= (number_columns-1)):
                memo_array[i][j] = min(memo_array[i-1][j-1],
                                       memo_array[i-1][j],
                                       memo_array[i-1][j+1]) + occupancy_probability[i][j]
            # If memo_array[i][j] only has value in memo_array (directly above it) and (to its to right and up).
            elif j+1 <= (number_columns - 1):
                memo_array[i][j] = min(memo_array[i-1][j],
                                       memo_array[i-1][j+1]) + occupancy_probability[i][j]
            # If memo_array[i][j] only has value in memo_array (directly above it) and (to its to left and up).
            else:
                memo_array[i][j] = min(memo_array[i-1][j-1],
                                       memo_array[i-1][j]) + occupancy_probability[i][j]

    output = reconstruct_solution(
        memo_array, number_rows, number_columns, occupancy_probability)
    return output