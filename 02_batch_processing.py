import numpy as np


def batch_processing_demo():
    """
    Demonstrates Matrix-Matrix Multiplication: Batch Processing in AI.
    Unit 1.3: Matrix Algebra.

    Scenario:
    Instead of 1 house, we have a 'batch' of 3 houses.
    We want to process all 3 simultaneously using the same Weight Matrix.
    """

    # 1. Define the Batch of Inputs (X)
    # We stack 3 input vectors as columns.
    # Shape: (2, 3) -> 2 features, 3 separate data samples (houses)
    input_batch = np.array([
        [0.8, 0.5, 0.9],  # Feature 1 (Area) for House 1, 2, 3
        [4.0, 2.0, 5.0]  # Feature 2 (Rooms) for House 1, 2, 3
    ])
    print(f"Input Batch shape: {input_batch.shape}")

    # 2. Define the Weight Matrix (W) - SAME as before
    # Shape: (3, 2) -> Transforms 2 features into 3 hidden states
    weights = np.array([
        [0.5, 0.1],
        [-0.2, 0.9],
        [0.8, -0.5]
    ])

    # 3. Perform Batch Processing (Matrix Multiplication)
    # Operation: Y = W * X
    # Shapes: (3,2) * (2,3) = (3,3)
    # The inner dimensions (2 and 2) match.
    output_batch = weights @ input_batch

    # 4. Output Result
    # Expected Shape: (3, 3) -> 3 output neurons for EACH of the 3 houses.
    print("-" * 30)
    print(f"Output Batch (Y):\n{output_batch}")
    print(f"Output Batch shape: {output_batch.shape}")

    # Verification for the first house (first column):
    # Should match the result from script 01 [0.8, 3.44, -1.36]
    first_house_output = output_batch[:, 0]
    print(f"\nFirst house output: {first_house_output}")

    # Standard Matrix Multiplication check (Row 1 * Col 1)
    # (0.5 * 0.8) + (0.1 * 4.0) = 0.4 + 0.4 = 0.8
    assert np.isclose(output_batch[0, 0], 0.8), "Calculation Error!"


if __name__ == "__main__":
    batch_processing_demo()