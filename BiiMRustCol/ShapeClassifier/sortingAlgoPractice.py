import numpy as np
from random import shuffle

def partition(nums, low, high):
    pivot = nums[(low + high) // 2]
    i = low - 1
    j = high + 1
    while True:
        i += 1
        while nums[i] < pivot:
            i += 1

        j -= 1
        while nums[j] > pivot:
            j -= 1

        if i >= j:
            return j

        # If an element at i (on the left of the pivot) is larger than the
        # element at j (on right right of the pivot), then swap them
        nums[i], nums[j] = nums[j], nums[i]

def quick_sort(nums):
    # Create a helper function that will be called recursively
    def _quick_sort(items, low, high):
        if low < high:
            # This is the index after the pivot, where our lists are split
            split_index = partition(items, low, high)
            _quick_sort(items, low, split_index)
            _quick_sort(items, split_index + 1, high)

    _quick_sort(nums, 0, len(nums) - 1)

random_list_of_nums = [22, 5, 1, 18, 99, 24, 105, 53, 69]
quick_sort(random_list_of_nums)
print(random_list_of_nums)



# Bubble sort implementation
random_list_of_nums = [22, 5, 1, 18, 99, 24, 105, 53, 69]
def bubbleSort(arr):
    trigger = True
    counter = 0
    while trigger == True:
        trigger = False
        counter += 1
        for i in range(0, len(arr)-counter):
            if arr[i] > arr[i+1]:
                holder = arr[i]
                arr[i] = arr[i+1]
                arr[i+1] = holder
                trigger = True
    return arr
print(bubbleSort(random_list_of_nums))


x = np.matrix([2,2])
# x = np.array([1,2])
# x = np.transpose(x)
y = np.matrix([2,2])
y = 3
print(y*x)
print(x)
print("*")
print(y)
print("----")
print(np.add(x, y))

z = np.matrix([[1,2,3],[1,2,3]])
for i in range(0,1):
    print(range(0, 1))