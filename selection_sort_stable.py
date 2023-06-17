def unstable_selection_sort(array):
    for i in range(len(array)):
        min = array[i]
        index = i
        for j in range(i+1, len(array)):
            if array[j] < min:
                min = array[j]
                index = j
        array[i], array[index] = array[index], array[i]
    return array

def unstable_selection_sort_with_tuples(array):
    for i in range(len(array)):
        array[i] = (array[i], i)
    for i in range(len(array)):
        min = array[i]
        index = i
        for j in range(i+1, len(array)):
            if array[j][0] < min[0]:
                min = array[j]
                index = j
        array[i], array[index] = array[index], array[i]
    return array

def stable_selection_sort(array):
    """
        Ensures the stability of our algorithm by swapping two identical values whilst being compared
        whenever they're out of original relative ordering
    """
    # Initializing array to contain tuples:
    for i in range(len(array)):
        array[i] = (array[i], i)
    for i in range(len(array)):
        min = array[i]
        index = i
        for j in range(i+1, len(array)):
            if array[j][0] < min[0]:
                min = array[j]
                index = j
            # Add extra conditional to check: If they're they're identical elements and indicies are out of place, swap:
            if (array[j][0] == min[0]) and (array[j][1] < min[1]):
                array[j], array[i] = array[i], array[j]
                min = array[i]

        array[i], array[index] = array[index], array[i]
    return array