import numpy as np
import matplotlib.pyplot as plt


def geometric_transformation_demo():
    """
    Demonstrates a Geometric Linear Mapping: Reflection.
    Unit 3.1: Matrix Representation of Linear Mappings.
    """

    # 1. Define the Reflection Matrix
    # This matrix flips the y-coordinate (x2)
    A = np.array([
        [1, 0],
        [0, -1]
    ])
    print(f"Transformation Matrix A:\n{A}")

    # 2. Define the Input Vector (Original Point)
    x = np.array([2, 3])
    print(f"\nInput Vector x: {x}")

    # 3. Apply the Transformation
    y = A @ x
    print(f"Output Vector y (A @ x): {y}")

    # 4. Verify the Math
    # y should be [2, -3]
    expected_y = np.array([2, -3])
    if np.array_equal(y, expected_y):
        print("\nSUCCESS: Calculation verified!")
    else:
        print("\nFAIL: Calculation mismatch.")

    # --- Optional: Visualization Logic ---
    # (This part creates a plot to SEE the reflection)
    origin = np.array([0, 0])

    plt.figure()
    # Plot Original Vector (Blue)
    plt.quiver(*origin, x[0], x[1], color='b', scale=1, scale_units='xy', angles='xy', label='Original')
    # Plot Transformed Vector (Red)
    plt.quiver(*origin, y[0], y[1], color='r', scale=1, scale_units='xy', angles='xy', label='Reflected')

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid()
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.title("Linear Mapping: Reflection across X-axis")
    plt.show() # Uncomment to display graph if running locally


if __name__ == "__main__":
    geometric_transformation_demo()
