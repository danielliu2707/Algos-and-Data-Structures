def radix_sort(input_array, base, no_digits):
    """
        Iterates over all the digit columns for my numbers and sort each column using counting sort.

        Time Complexity: No. Digits * Comp(Counting Sort)
        Best-Case: O(N) where N is the size of array
        Worst-Case: O(log_b(M) * (N + b)) where M is the max length of digits of a number

        Space Complexity: No. digits * Comp(Counting Sort)
        Same as above.
    """
    for i in range(no_digits):
        input_array = radix_pass(input_array, base, i)
    return input_array

def radix_pass(input_array, base, digit):
    # Initializing count and position arrays of fixed size
    count_array = [0 for i in range(base)]
    position_array = [0 for i in range(base)]
    # Populating count_array
    for element in input_array:
        count_array[get_digit(element, base, digit)] += 1
    # Populating position_array
    position_array[0] = 1
    for i in range(1, base):
        position_array[i] = position_array[i-1] + count_array[i-1]
    # Go through input_array to populate output_array
    output_array = [None] * len(input_array)
    for element in input_array:
        output_array[position_array[get_digit(
            element, base, digit)] - 1] = element
        position_array[get_digit(
            element, base, digit)] += 1
    return output_array


def get_digit(element: int, base, digit):
    """
        Obtains the appropriate digit from an integer element in base (b).
        Note: Digit should start at index = 0.
    """
    return element // base**digit % 10