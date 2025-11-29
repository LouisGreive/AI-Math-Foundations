import numpy as np

def affine_mapping_demo():
    """
    Demonstrates an Affine Mapping: The Math of a Neural Network Layer.
    Formula: y = Wx + b
    Unit 3.4: Affine Mappings.
    """

    # 1. Define the Weights Matrix (Linear Part A)
    # Using the values from our manual exercise
    weights = np.array([
        [2, 3],
        [1, 2]
    ])
    print(f"Weights Matrix (W):\n{weights}")

    # 2. Define the Bias Vector (Translation Part b)
    bias = np.array([-2, 2])
    print(f"\nBias Vector (b): {bias}")

    # 3. Define the Input Vector (x)
    input_vector = np.array([1, 3])
    print(f"\nInput Vector (x): {input_vector}")

    # 4. Perform the Affine Mapping (Forward Pass)
    # Step A: Linear Transformation (Rotation/Scaling)
    linear_part = weights @ input_vector
    print(f"Step 1 (W @ x): {linear_part}")  # Should be [11, 7]

    # Step B: Translation (Adding Bias)
    output_vector = linear_part + bias
    print(f"Step 2 (W @ x + b): {output_vector}") # Should be [9, 9]

    # 5. Verification
    expected_output = np.array([9, 9])
    if np.array_equal(output_vector, expected_output):
        print("\nSUCCESS: Artificial Neuron calculation verified!")
    else:
        print("\nFAIL: Calculation mismatch.")

if __name__ == "__main__":
    affine_mapping_demo()