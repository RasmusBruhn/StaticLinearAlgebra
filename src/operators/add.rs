use crate::Matrix;
use crate::LinAlgError;

pub fn matrix<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Result<Matrix<T>, LinAlgError>
    where
    T: Clone,
    T: Copy,
    T: std::ops::Add<Output = T>,
    T: std::ops::Mul<Output = T>,
    T: std::iter::Sum,
{
    if lhs.size != rhs.size {
        return Err(LinAlgError::MatchDimensions(format!("Matrices must have identical size when adding, received {:?} and {:?}", lhs.size, rhs.size)));
    }

    let size = lhs.size;
    let values = (0..size.0*size.1).map(|i| lhs.values[i] + rhs.values[i]).collect();
    Ok(Matrix::from_vec(values, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fn_matrix() {
        let matrix1 = Matrix::from_arr(&[0, 1, 2, 3, 4, 5], (3, 2));
        let matrix2 = Matrix::from_arr(&[0, 10, 20, 30, 40, 50], (3, 2));
        let result = matrix(&matrix1, &matrix2).unwrap();
        assert_eq!((3, 2), result.get_size());
        assert_eq!(vec![0, 11, 22, 33, 44, 55], result.to_vec());
    }

    #[test]
    fn fn_matrix_err() {
        let matrix1 = Matrix::from_arr(&[0, 1, 2, 3, 4, 5], (2, 3));
        let matrix2 = Matrix::from_arr(&[0, 10, 20, 30, 40, 50], (3, 2));
        assert!(matrix(&matrix1, &matrix2).is_err());
    }
}