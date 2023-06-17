# Counting Sort Algorithm:

def counting_sort(input_array):
    """
        The following implement Stable Counting Sort, an non-comparison based sorting algorithm used to run with good complexity.

        Let N be the size of the input array.
        Let U be the size of the count/position arrays.

        Time Complexity:
        BestCase: O(N + N + N + U) = O(N) when N = U
        WorstCase: O(N + N + N + U) = O(N + U)

        Space Complexity:
        O(N + 2U) = O(N + U) as you have output array (N) and count/position array (U)
    """
    # Finding max element
    max_element = input_array[0]
    for element in input_array:
        if element > max_element:
            max_element = element
    # Initializing count and position arrays of fixed size
    count_array = [0 for i in range(max_element)]
    position_array = [0 for i in range(max_element)]
    # Populating count_array
    for element in input_array:
        count_array[element-1] += 1
    # Populating position_array
    position_array[0] = 1
    for i in range(1, max_element):
        position_array[i] = position_array[i-1] + count_array[i-1]
    # Go through input_array to populate output_array
    output_array = [None] * len(input_array)
    for element in input_array:
        output_array[position_array[element-1] - 1] = element
        position_array[element-1] += 1
    return output_array