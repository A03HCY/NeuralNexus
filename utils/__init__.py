import math


def calculate_layer(step:int, kernel_size:int=3):
    if kernel_size <= 1:
        raise ValueError("kernel_size must be greater than 1")
    L = math.ceil(math.log2((step - 1) / (kernel_size - 1) + 1))
    R = 1 + (kernel_size - 1) * (2 ** L - 1)
    return int(L), R

print(calculate_layer(100))