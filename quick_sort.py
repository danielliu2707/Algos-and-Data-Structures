# Write a function which implements QuickSort:

def quick_sort(array, left, right):
    """
        The following function implements QuickSort using a randomised pivot
    """
    if left < right:
        pivot = array[left]
        # Partition returns the index (q) of the pivot
        (b_counter, r_counter) = partition(array, left, right, pivot)
        # Perform quicksort on left_subarray
        quick_sort(array, left, b_counter - 1)
        # Perform quicksort on right_subarray
        quick_sort(array, r_counter + 1, right)
    return array


def partition(array, left, right, pivot):
    """
        DNF Partitioning Algorithm:
    """
    blue_counter = left
    red_counter = right
    j = left
    while j <= red_counter:
        if array[j] < pivot:
            # Swap the item at array[j] with blue_counter:
            array[j], array[blue_counter] = array[blue_counter], array[j]
            blue_counter += 1
            j += 1
        elif array[j] > pivot:
            array[j], array[red_counter] = array[red_counter], array[j]
            red_counter -= 1
        else:
            j += 1
    return (blue_counter, red_counter)


# Write a function which Implements Quick Select with a random pivot choice:

def quick_select(array, k, lo, hi):
    """
        Returns the k'th smallest element within the array

        Note: It's (k-1) because indicies start at 0
    """
    if len(array) > 0:
        pivot = array[lo]
        (mid, _) = partition(array, lo, hi, pivot)
        if (k-1) < (mid):
            return quick_select(array, k, lo, mid-1)
        elif (k-1) > (mid):
            return quick_select(array, k, mid+1, hi)
        else:
            return array[k-1]
    else:
        return

# Modify Quick sort so that it uses the Quick select to choose the median element:

def quick_sort_median(array, left, right):
    """
        The following function implements QuickSort using the median pivot everytime
    """
    if left < right:
        median_index = (len(array) + 1) // 2
        pivot = quick_select(array, median_index)
        # Partition returns the index (q) of the pivot
        (b_counter, r_counter) = partition(array, left, right, pivot)
        # Perform quicksort on left_subarray
        quick_sort(array, left, b_counter - 1)
        # Perform quicksort on right_subarray
        quick_sort(array, r_counter + 1, right)
    return array
