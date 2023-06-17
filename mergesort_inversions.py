def merge_sort_inversions(array):
    if len(array) == 1:
        return (array, 0)
    else:
        mid_point = len(array) // 2

        (left_partition, no_inv) = merge_sort_inversions(array[:mid_point])
        (right_partition, no_inv_2) = merge_sort_inversions(array[mid_point:])
        total_inv = no_inv + no_inv_2
        return merge(left_partition, right_partition, total_inv)


def merge(l_array, r_array, no_inv):
    tmp_lst = []
    i = j = 0

    while i < len(l_array) and j < len(r_array):
        if l_array[i] <= r_array[j]:
            tmp_lst.append(l_array[i])
            i += 1
        else:
            tmp_lst.append(r_array[j])
            j += 1
            no_inv += len(l_array[i:])
    tmp_lst.extend(l_array[i:])
    tmp_lst.extend(r_array[j:])
    return (tmp_lst, no_inv)