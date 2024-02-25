# Version 0

## Update 0.2.0

Matrices:
- Added method "identity" to create the identity matrix
- Added trait "One"
- Added trait "Product" for matrices and references
- Fixed trait "AddAssign" such that it cannot change the type of the original value
- Fixed trait "SubAssign" such that it cannot change the type of the original value
- Fixed trait "MulAssign" such that it cannot change the type of the original value
- Removed left side multiplication
- TODO: Add determinant
- TODO: Add inverse

Column vectors:
- Fixed trait "AddAssign" such that it cannot change the type of the original value
- Fixed trait "SubAssign" such that it cannot change the type of the original value
- Fixed trait "MulAssign" such that it cannot change the type of the original value
- Removed left side multiplication

Row vectors:
- Fixed trait "AddAssign" such that it cannot change the type of the original value
- Fixed trait "SubAssign" such that it cannot change the type of the original value
- Fixed trait "MulAssign" such that it cannot change the type of the original value
- Removed left side multiplication

## Update 0.1.0

Added matrices of any copyable type:
- Added method "new" to create a new matrix
- Added method "from_value" to create a matrix filled with a single value
- Added method "get_values" to get a reference to the values
- Added method "get_values_mut" to get a mutable reference to the values
- Added method "transpose" to transpose a matrix returning the transposed one
- Added method "from_diag" to create a diagonal matrix, requires the type to be zeroable
- Added method "hermitian_conjugate" to get the hermitian conjugate of a complex matrix
- Added method "is_symmetric" to check if a square matrix is symmetric
- Added method "is_hermitian" to check if a square complex matrix is hermitian
- Added traits "Index" and "IndexMut"
- Added trait "Zero"
- Added trait "Sum" for matrices and references
- Added traits "Add" and "AddAssign"
- Added traits "Sub" and "SubAssign"
- Added traits "Mul" and "MulAssign" with matrices
- Added traits "Mul" and "MulAssign" with scalars
- Added trait "Mul" with column vectors
- Added trait "Mul" to scalars for matrices for left side multiplication

Added column vectors of any copyable type:
- Added method "new" to create a new vector
- Added method "from_value" to create a vector filled with a single value
- Added method "get_values" to get a reference to the values
- Added method "get_values_mut" to get a mutable reference to the values
- Added method "transpose" to transpose a columns vector returning the transposed row vector
- Added method "hermitian_conjugate" to get the hermitian conjugate of a complex column vector
- Added traits "Index" and "IndexMut"
- Added trait "Zero"
- Added trait "Sum" for column vectors and references
- Added traits "Add" and "AddAssign"
- Added traits "Sub" and "SubAssign"
- Added traits "Mul" and "MulAssign" with scalars
- Added trait "Mul" with column vectors (inner product)
- Added trait "Mul" with row vectors (outer product)
- Added trait "Mul" to scalars for column vectors for left side multiplication

Added row vectors of any copyable type:
- Added method "new" to create a new vector
- Added method "from_value" to create a vector filled with a single value
- Added method "get_values" to get a reference to the values
- Added method "get_values_mut" to get a mutable reference to the values
- Added method "transpose" to transpose a row vector returning the transposed column vector
- Added method "hermitian_conjugate" to get the hermitian conjugate of a complex row vector
- Added traits "Index" and "IndexMut"
- Added trait "Zero"
- Added trait "Sum" for row vectors and references
- Added traits "Add" and "AddAssign"
- Added traits "Sub" and "SubAssign"
- Added traits "Mul" and "MulAssign" with scalars
- Added trait "Mul" with row vectors (inner product)
- Added trait "Mul" with column vectors (inner product)
- Added trait "Mul" with matrices
- Added trait "Mul" to scalars for row vectors for left side multiplication
