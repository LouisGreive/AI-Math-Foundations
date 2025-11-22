import numpy as np

def print_vector_info(name, vector):
    print(f"--- {name} ---")
    print(f"Vector: {vector}")
    print(f"Shape:  {vector.shape}")
    print(f"Type:   {vector.dtype}\n")

# 1. Defining Vectors
# A vector in AI is just an ordered list of numbers (like coordinates)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print_vector_info("Vector 1", v1)
print_vector_info("Vector 2", v2)

# 2. Vector Addition
# In Logistics: Adding two inventory shipments together
v_total = v1 + v2
print(f"Vector Addition (v1 + v2): {v_total}\n")

# 3. Scalar Multiplication
# In Logistics: Doubling an order
v_scaled = v1 * 2
print(f"Scalar Multiplication (v1 * 2): {v_scaled}\n")

# 4. The Dot Product (The "similarity" score)
# This is the engine of many AI algorithms
dot_product = np.dot(v1, v2)
print(f"Dot Product (v1 . v2): {dot_product}")
# Calculation: (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32