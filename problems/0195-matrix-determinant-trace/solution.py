def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
    """
    Compute the determinant and trace of a square matrix.
    
    Args:
        matrix: A square matrix (n x n) represented as list of lists
    
    Returns:
        Tuple of (determinant, trace)
    """
    n = len(matrix)

    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be a non-empty square matrix")
    
    trace = sum(matrix[i][i] for i in range(n))

    mat = [row[:] for row in matrix]
    determinant = 1.0

    for i in range(n):

        pivot = i

        for r in range(i + 1, n):
            if abs(mat[r][i]) > abs(mat[pivot][i]):
                pivot = r
        
        # If pivot is zero, determinant is zero
        if mat[pivot][i] == 0:
            return 0.0, float(trace)

        if pivot != i:
            mat[i], mat[pivot] = mat[pivot], mat[i]
            determinant *= -1
        
        # Multiply determinant by pivot
        pivot_value = mat[i][i]
        determinant *= pivot_value

        # Eliminate elements below the pivot
        for j in range(i + 1, n):
            factor = mat[j][i] / pivot_value

            for k in range(i + 1, n):
                mat[j][k] -= factor * mat[i][k]
        
    return float(determinant), float(trace)

