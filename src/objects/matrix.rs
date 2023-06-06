use std::ops::{Index, IndexMut, Add, Sub, Mul};
use num::traits::Zero;
use itertools::Itertools;
use std::iter::Sum;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Matrix<T, const R: usize, const C: usize>
where
    T: Copy,
{
    pub(crate) values: [[T; C]; R],
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Copy,
{
    pub fn new(values: &[[T; C]; R]) -> Self {
        Self {values: *values}
    }

    pub fn from_value(value: T) -> Self {
        Self {values: [[value; C]; R]}
    }

    pub fn get_values(&self) -> &[[T; C]; R] {
        &self.values
    }

    pub fn get_values_mut(&mut self) -> &mut [[T; C]; R] {
        &mut self.values
    }
}

impl<T, const S: usize> Matrix<T, S, S>
where
    T: Copy,
    T: Zero,
{
    pub fn from_diag(values: &[T; S]) -> Self {
        let mut use_values= [[T::zero(); S]; S];
        
        for (n, value) in values.iter().enumerate() {
            use_values[n][n] = *value;
        }

        Self {values: use_values}
    }
}

impl<T, const R: usize, const C: usize> Index<usize> for Matrix<T, R, C> 
where
    T: Copy,
{
    type Output = [T; C];

    fn index(&self, idx: usize) -> &Self::Output {
        &self.values[idx]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<usize> for Matrix<T, R, C> 
where
    T: Copy,
{
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.values[idx]
    }
}

impl<T, const R: usize, const C: usize> Zero for Matrix<T, R, C>
where
    T: Copy,
    T: Zero,
    T: PartialEq,
{
    fn zero() -> Self {
        Self::from_value(T::zero())
    }

    fn is_zero(&self) -> bool {
        (0..R).cartesian_product(0..C).any(|(r, c)| self[r][c] != T::zero()) ^ true
    }
}

impl<T, const R: usize, const C: usize> Sum for Matrix<T, R, C>
where
    T: Copy,
    T: Zero,
    T: Add<T, Output = T>,
    T: PartialEq,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut result: Self = Matrix::zero();

        for value in iter {
            result = result + value;
        }

        result
    }
}

impl<'a, T, const R: usize, const C: usize> Sum<&'a Matrix<T, R, C>> for Matrix<T, R, C>
where
    T: Copy,
    T: Zero,
    T: Add<T, Output = T>,
    T: PartialEq,
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let mut result: Self = Matrix::zero();

        for value in iter {
            result = result + *value;
        }

        result
    }
}

impl<T, TR, O, const R: usize, const C: usize> Add<Matrix<TR, R, C>> for Matrix<T, R, C>
where
    T: Copy,
    T: Add<TR, Output = O>,
    TR: Copy,
    O: Copy,
{
    type Output = Matrix<O, R, C>;

    fn add(self, rhs: Matrix<TR, R, C>) -> Self::Output {
        let values: [[O; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| self[r][c] + rhs[r][c]).collect::<Vec<O>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[O; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<T, TR, O, const R: usize, const C: usize> Add<&Matrix<TR, R, C>> for &Matrix<T, R, C>
where
    T: Copy,
    for<'a> &'a T: Add<&'a TR, Output = O>,
    TR: Copy,
    O: Copy,
{
    type Output = Matrix<O, R, C>;

    fn add(self, rhs: &Matrix<TR, R, C>) -> Self::Output {
        let values: [[O; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| &self[r][c] + &rhs[r][c]).collect::<Vec<O>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[O; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<T, TR, O, const R: usize, const C: usize> Sub<Matrix<TR, R, C>> for Matrix<T, R, C>
where
    T: Copy,
    T: Sub<TR, Output = O>,
    TR: Copy,
    O: Copy,
{
    type Output = Matrix<O, R, C>;

    fn sub(self, rhs: Matrix<TR, R, C>) -> Self::Output {
        let values: [[O; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| self[r][c] - rhs[r][c]).collect::<Vec<O>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[O; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<T, TR, O, const R: usize, const C: usize> Sub<&Matrix<TR, R, C>> for &Matrix<T, R, C>
where
    T: Copy,
    for<'a> &'a T: Sub<&'a TR, Output = O>,
    TR: Copy,
    O: Copy,
{
    type Output = Matrix<O, R, C>;

    fn sub(self, rhs: &Matrix<TR, R, C>) -> Self::Output {
        let values: [[O; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| &self[r][c] - &rhs[r][c]).collect::<Vec<O>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[O; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<T, TR, O, const R: usize, const K: usize, const C: usize> Mul<Matrix<TR, K, C>> for Matrix<T, R, K>
where
    T: Copy,
    T: Mul<TR, Output = O>,
    TR: Copy,
    O: Copy,
    O: Sum,
{
    type Output = Matrix<O, R, C>;

    fn mul(self, rhs: Matrix<TR, K, C>) -> Self::Output {
        let values: [[O; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| 
            (0..K).map(|k| self[r][k] * rhs[k][c]).sum()).collect::<Vec<O>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[O; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<T, TR, O, const R: usize, const K: usize, const C: usize> Mul<&Matrix<TR, K, C>> for &Matrix<T, R, K>
where
    T: Copy,
    for<'a> &'a T: Mul<&'a TR, Output = O>,
    TR: Copy,
    O: Copy,
    O: Sum,
{
    type Output = Matrix<O, R, C>;

    fn mul(self, rhs: &Matrix<TR, K, C>) -> Self::Output {
        let values: [[O; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| 
            (0..K).map(|k| &self[r][k] * &rhs[k][c]).sum()).collect::<Vec<O>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[O; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod matrix {
        use super::*;

        #[test]
        fn new() {
            let result = Matrix::new(&[[0, 1], [2, 3], [4, 5]]);
            assert_eq!([[0, 1], [2, 3], [4, 5]], result.values);
        }

        #[test]
        fn from_value() {
            let result: Matrix<f32, 2, 5> = Matrix::from_value(0.);
            assert_eq!([[0.; 5]; 2], result.values);
        }

        #[test]
        fn from_diag() {
            let result = Matrix::from_diag(&[1, 2]);
            assert_eq!([[1, 0], [0, 2]], result.values);
        }

        #[test]
        fn get_values() {
            let result = Matrix::new(&[[0, 1, 2], [3, 4, 5]]);
            assert_eq!(&[[0, 1, 2], [3, 4, 5]], result.get_values());
        }

        #[test]
        fn get_values_mut() {
            let mut result = Matrix::new(&[[0, 1, 2], [3, 4, 5]]);
            let data = result.get_values_mut();
            data[0][1] = 10;
            assert_eq!([[0, 10, 2], [3, 4, 5]], result.values);
        }

        #[test]
        fn matrix_of_matrix() {
            let result = Matrix::new(&[[Matrix::new(&[[1, 0], [0, 1]])], [Matrix::new(&[[0, 1], [-1, 0]])]]);
            assert_eq!([[Matrix::new(&[[1, 0], [0, 1]])], [Matrix::new(&[[0, 1], [-1, 0]])]], result.values);
        }

        #[test]
        fn index_get() {
            let result = Matrix::new(&[[0, 1, 2], [3, 4, 5]]);
            assert_eq!(2, result[0][2]);
        }

        #[test]
        fn index_set() {
            let mut result = Matrix::new(&[[0, 1, 2], [3, 4, 5]]);
            result[0][2] = 10;
            assert_eq!(10, result[0][2]);
        }

        #[test]
        fn zero() {
            let matrix_zero: Matrix<f64, 4, 5> = Matrix::zero();
            assert_eq!([[0f64; 5]; 4], matrix_zero.values);
        }

        #[test]
        fn is_zero_true() {
            let result: Matrix<i32, 3, 3> = Matrix::from_value(0);
            assert_eq!(true, result.is_zero());
        }

        #[test]
        fn is_zero_false() {
            let mut result: Matrix<i32, 3, 3> = Matrix::from_value(1);
            result[1][0] = 1;
            assert_eq!(false, result.is_zero());
        }

        #[test]
        fn sum() {
            let list: [Matrix<i32, 2, 2>; 3] = [Matrix::new(&[[0, 1], [2, 3]]), Matrix::new(&[[0, 10], [20, 30]]), Matrix::new(&[[0, 100], [200, 300]])];
            let result: Matrix<i32, 2, 2> = list.into_iter().sum();
            assert_eq!([[0, 111], [222, 333]], result.values);
        }

        #[test]
        fn sum_ref() {
            let list: [Matrix<i32, 2, 2>; 3] = [Matrix::new(&[[0, 1], [2, 3]]), Matrix::new(&[[0, 10], [20, 30]]), Matrix::new(&[[0, 100], [200, 300]])];
            let result: Matrix<i32, 2, 2> = list.iter().sum();
            assert_eq!([[0, 111], [222, 333]], result.values);
        }

        #[test]
        fn add() {
            let a = Matrix::new(&[[0, 1], [2, 3]]);
            let b = Matrix::new(&[[0, 10], [20, 30]]);
            let c = a + b;
            assert_eq!([[0, 11], [22, 33]], c.values);
        }

        #[test]
        fn add_ref() {
            let a = Matrix::new(&[[0, 1], [2, 3]]);
            let b = Matrix::new(&[[0, 10], [20, 30]]);
            let c = &a + &b;
            assert_eq!([[0, 11], [22, 33]], c.values);
        }

        #[test]
        fn sub() {
            let a = Matrix::new(&[[0, 1], [2, 3]]);
            let b = Matrix::new(&[[0, 10], [20, 30]]);
            let c = a - b;
            assert_eq!([[0, -9], [-18, -27]], c.values);
        }

        #[test]
        fn sub_ref() {
            let a = Matrix::new(&[[0, 1], [2, 3]]);
            let b = Matrix::new(&[[0, 10], [20, 30]]);
            let c = &a - &b;
            assert_eq!([[0, -9], [-18, -27]], c.values);
        }

        #[test]
        fn mul() {
            let matrix1 = Matrix::new(&[[0, 1, 2, 3], [4, 5, 6, 7]]);
            let matrix2 = Matrix::new(&[[0, 10, 20], [30, 40, 50], [60, 70, 80], [90, 100, 110]]);
            let result = matrix1 * matrix2;
            assert_eq!([[420, 480, 540], [1140, 1360, 1580]], result.values);    
        }

        #[test]
        fn mul_ref() {
            let matrix1 = Matrix::new(&[[0, 1, 2, 3], [4, 5, 6, 7]]);
            let matrix2 = Matrix::new(&[[0, 10, 20], [30, 40, 50], [60, 70, 80], [90, 100, 110]]);
            let result = &matrix1 * &matrix2;
            assert_eq!([[420, 480, 540], [1140, 1360, 1580]], result.values);    
        }
    }
}
