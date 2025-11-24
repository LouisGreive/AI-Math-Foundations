import numpy as np

def mapping_verification():
    """
    Verifies that our calculated Matrix A performs the same
    linear mapping as the algebraic function f(x).
    Unit 3.1: Matrix Representation of Linear Mappings.
    """

    # 1. Define the calculated Matrix A
    # We found this by mapping the basis vectors [1,0] and [0,1]
    A = np.array([
        [3, 2],
        [1, 0]
    ])
    print(f"Matrix A:\n{A}")

    # 2. Define a random test vector x
    # Let's pick x1=2, x2=5 to test if it works
    x = np.array([2, 5])
    print(f"\nTest Vector x: {x}")

    # 3. Method 1: Use the Matrix (A @ x)
    # This is how a neural network layer would do it
    y_matrix = A @ x
    print(f"Result using Matrix (A @ x): {y_matrix}")

    # 4. Method 2: Use the Function Formula f(x)
    # f(x) = [2*x2 + 3*x1, x1]
    # y[0] = 2*5 + 3*2 = 10 + 6 = 16
    # y[1] = 2
    y_formula = np.array([
        2 * x[1] + 3 * x[0],
        x[0]
    ])
    print(f"Result using Formula f(x):   {y_formula}")

    # 5. Verify they are identical
    if np.array_equal(y_matrix, y_formula):
        print("\nSUCCESS: The matrix correctly represents the linear mapping!")
    else:
        print("\nFAIL: The matrix does not match the function.")

if __name__ == "__main__":
    mapping_verification()