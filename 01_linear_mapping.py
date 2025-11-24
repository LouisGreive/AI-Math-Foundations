import numpy as np


def linear_mapping_demo():
    """
    Demonstrates Matrix-Vector Multiplication: The core of a Neural Network Layer.
    Unit 3: Linear Mappings.

    Scenario:
    We have an input vector representing a house with 2 features:
    - Feature 1: Area (normalized)
    - Feature 2: Number of Rooms

    We want to transform this into a 'hidden state' of 3 neurons using a Weight Matrix.
    """

    # 1. Define the Input Vector (x)
    # Shape: (2,) representing 2 input features
    input_vector = np.array([0.8, 4])
    print(f"Input Vector shape: {input_vector.shape}")

    # 2. Define the Transformation Matrix (A), often called 'Weights' (W) in AI
    # Shape: (3, 2) -> We want to transform 2 inputs into 3 outputs
    # Rows = number of output neurons (3)
    # Cols = number of input features (2)
    # In Linear Algebra terms: Mapping from R^2 to R^3
    weight_matrix = np.array([
        [0.5, 0.1],  # Neuron 1 weights
        [-0.2, 0.9],  # Neuron 2 weights
        [0.8, -0.5]  # Neuron 3 weights
    ])
    print(f"Weight Matrix shape: {weight_matrix.shape}")

    # 3. Perform the Linear Mapping (Matrix-Vector Multiplication)
    # Mathematical operation: y = A * x
    # In NumPy, we use the '@' operator for matrix multiplication (dot product)
    output_vector = weight_matrix @ input_vector

    # 4. Output Result
    # Expected Shape: (3,) -> 3 output neurons (a vector in R^3)
    print("-" * 30)
    print(f"Output Vector (y): {output_vector}")
    print(f"Output shape: {output_vector.shape}")

    # Verification of calculation for the first element:
    # y[0] = (Row 1 of Matrix) dot (Input Vector)
    # y[0] = (0.5 * 0.8) + (0.1 * 4) = 0.4 + 0.4 = 0.8
    assert np.isclose(output_vector[0], 0.8), "Calculation Error!"


if __name__ == "__main__":
    linear_mapping_demo()