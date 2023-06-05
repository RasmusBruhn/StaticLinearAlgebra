use crate::Matrix;
use crate::LinAlgError;
use itertools::Itertools;

pub fn matrix_matrix<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Result<Matrix<T>, LinAlgError>
    where
    T: Clone,
    T: std::ops::Add<Output = T>,
    T: std::ops::Mul<Output = T>,
    T: std::iter::Sum,
{
    if lhs.size.1 != rhs.size.0 {
        return Err(LinAlgError::MatchDimensions(format!("Matrices must have size (n x m) * (m x k) when multiplying, received {:?} and {:?}", lhs.size, rhs.size)));
    }

    let size = (lhs.size.0, rhs.size.1);
    let values = (0..size.0).cartesian_product(0..size.1)
    .map(|(i, j)| (0..lhs.size.1)
    .map(|k| lhs.values[k + i * lhs.size.1].clone() * rhs.values[j + rhs.size.1 * k].clone()).sum()).collect();
    Ok(Matrix::from_vec(values, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fn_matrix_matrix() {
        let matrix1 = Matrix::from_arr(&[0, 1, 2, 3, 4, 5, 6, 7], (2, 4));
        let matrix2 = Matrix::from_arr(&[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], (4, 3));
        let result = matrix_matrix(&matrix1, &matrix2).unwrap();
        assert_eq!((2, 3), result.get_size());
        assert_eq!(vec![420, 480, 540, 1140, 1360, 1580], result.to_vec());
    }

    #[test]
    fn fn_matrix_matrix_err() {
        let matrix1 = Matrix::from_arr(&[0, 1, 2, 3, 4, 5, 6, 7], (2, 4));
        let matrix2 = Matrix::from_arr(&[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], (3, 4));
        assert!(matrix_matrix(&matrix1, &matrix2).is_err());
    }
}