import numpy as np

def check_image_dimension():
    """
    Checks the dimension of the Image of Matrix A.
    Unit 3: Image and Kernel.
    """
    # 1. Define the Matrix from the exercise
    A = np.array([
        [ 0, -3,  2],
        [ 1,  2,  0],
        [-3,  3,  0]
    ])
    print(f"Matrix A:\n{A}")

    # 2. Calculate the Rank
    # The Rank tells us the number of linearly independent columns.
    # It corresponds to the dimension of the Image.
    rank = np.linalg.matrix_rank(A)
    print(f"\nRank of A: {rank}")

    # 3. Calculate the Determinant (The "Volume" check)
    det = np.linalg.det(A)
    print(f"Determinant of A: {det:.2f}")

    # 4. Interpretation
    if rank == 3:
        print("\nCONCLUSION: The vectors are Independent.")
        print("The Image spans the entire 3D space (R^3).")
    else:
        print("\nCONCLUSION: The vectors are Dependent.")
        print(f"The Image is collapsed into a {rank}D subspace.")

if __name__ == "__main__":
    check_image_dimension()
