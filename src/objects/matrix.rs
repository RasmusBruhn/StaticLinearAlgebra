//! Implementation and all methods on matrices

use std::ops::{Index, IndexMut, Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
use num::{traits::{Zero, Num}, Complex};
use itertools::Itertools;
use std::iter::Sum;
use core::ops::Neg;
use super::vector_column::VectorColumn;

/// A static matrix type
/// 
/// Size must be known at compile time but operations are checked for size compatibility at compile time too
/// 
/// R: The number of rows
/// 
/// C: The number of columns
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
    /// Initializes a new matrix with the given values
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new(&[[0, 1, 2], [3, 4, 5]]);
    /// 
    /// assert_eq!(&[[0, 1, 2], [3, 4, 5]], x.get_values());
    /// ```
    pub fn new(values: &[[T; C]; R]) -> Self {
        Self {values: *values}
    }

    /// Initializes a new matrix filled with a single value
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::<f32, 2, 2>::from_value(1.);
    /// 
    /// assert_eq!(&[[1., 1.], [1., 1.]], x.get_values());
    /// ```
    pub fn from_value(value: T) -> Self {
        Self {values: [[value; C]; R]}
    }

    /// Retrieves a reference to the data of the matrix
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new(&[[0, 1], [2, 3]]);
    /// let data = x.get_values();
    /// 
    /// assert_eq!(&[[0, 1], [2, 3]], data);
    /// ```
    pub fn get_values(&self) -> &[[T; C]; R] {
        &self.values
    }

    /// Retrieves a mutable reference to the data of the matrix
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = static_linear_algebra::Matrix::new(&[[0, 1], [2, 3]]);
    /// let data = x.get_values_mut();
    /// data[0][1] = 5;
    /// 
    /// assert_eq!(&[[0, 5], [2, 3]], x.get_values());
    /// ```
    pub fn get_values_mut(&mut self) -> &mut [[T; C]; R] {
        &mut self.values
    }

    /// Transposes the matrix, switching row and columns
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new(&[[0, 1, 2], [3, 4, 5]]);
    /// let y = x.transpose();
    /// 
    /// assert_eq!(&[[0, 3], [1, 4], [2, 5]], y.get_values());
    /// ```
    pub fn transpose(&self) -> Matrix<T, C, R> {
        let values: [[T; R]; C] = 
            match (0..C).map(|r| 
            match (0..R).map(|c| self[c][r]).collect::<Vec<T>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[T; R]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Matrix {values}
    }
}

impl<T, const S: usize> Matrix<T, S, S>
where
    T: Copy,
    T: Zero,
{
    /// Initializes a diagonal matrix where the diagonal contains the values given
    /// and everything else is 0
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::from_diag(&[2, 3]);
    /// 
    /// assert_eq!(&[[2, 0], [0, 3]], x.get_values());
    /// ```
    pub fn from_diag(values: &[T; S]) -> Self {
        let mut use_values= [[T::zero(); S]; S];
        
        for (n, value) in values.iter().enumerate() {
            use_values[n][n] = *value;
        }

        Self {values: use_values}
    }
}

impl<T, const R: usize, const C: usize> Matrix<Complex<T>, R, C> 
where
    T: Copy,
    T: Num,
    T: Neg<Output = T>,
{
    /// Takes the hermitian conjugate of the matrix (transpose the matrix 
    /// and complex conjugate each element (change the sign of the imaginary part))
    /// 
    /// # Examples
    /// 
    /// ```
    /// use num::Complex;
    /// 
    /// let x = static_linear_algebra::Matrix::new(&[[Complex::new(1, 0), Complex::new(0, 2)], [Complex::new(0, 3), Complex::new(0, 4)]]);
    /// let y = x.hermitian_conjugate();
    /// 
    /// assert_eq!(&[[Complex::new(1, 0), Complex::new(0, -3)], [Complex::new(0, -2), Complex::new(0, -4)]], y.get_values())
    /// ```
    pub fn hermitian_conjugate(&self) -> Matrix<Complex<T>, C, R> {
        let values: [[Complex<T>; R]; C] = 
            match (0..C).map(|r| 
            match (0..R).map(|c| self[c][r].conj()).collect::<Vec<Complex<T>>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[Complex<T>; R]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Matrix {values}
    }
}

impl<T, const S: usize> Matrix<T, S, S> 
where
    T: Copy,
    T: PartialEq,
{
    /// Checks if the matrix is symmetric (the matrix is equal to its own transpose)
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new(&[[0, 1], [1, 2]]);
    /// 
    /// assert_eq!(true, x.is_symmetric());
    /// ```
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new(&[[0, 1], [2, 1]]);
    /// 
    /// assert_eq!(false, x.is_symmetric());
    /// ```
    pub fn is_symmetric(&self) -> bool {
        (0..S).any(|r| (0..r+1).any(|c| self[r][c] != self[c][r])) ^ true
    }
}

impl<T, const S: usize> Matrix<Complex<T>, S, S> 
where
    T: Copy,
    T: Num,
    T: Neg<Output = T>,
{
    /// Checks if the matrix is hearmitian (the matrix is equal to its own hearmitian conjugate)
    /// 
    /// # Examples
    /// 
    /// ```
    /// use num::Complex;
    /// 
    /// let x = static_linear_algebra::Matrix::new(&[[Complex::new(0, 0), Complex::new(0, 1)], [Complex::new(0, -1), Complex::new(2, 0)]]);
    /// 
    /// assert_eq!(true, x.is_hermitian());
    /// ```
    /// 
    /// ```
    /// use num::Complex;
    /// 
    /// let x = static_linear_algebra::Matrix::new(&[[Complex::new(0, 0), Complex::new(0, 1)], [Complex::new(0, 1), Complex::new(2, 0)]]);
    /// 
    /// assert_eq!(false, x.is_hermitian());
    /// ```
    pub fn is_hermitian(&self) -> bool {
        (0..S).any(|r| (0..r+1).any(|c| self[r][c] != self[c][r].conj())) ^ true
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

impl<TL, TR, TO, const R: usize, const C: usize> Add<Matrix<TR, R, C>> for Matrix<TL, R, C>
where
    TL: Copy,
    TL: Add<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
{
    type Output = Matrix<TO, R, C>;

    /// Normal elementwise addition of two matrices
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new(&[[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::Matrix::new(&[[0, 10], [20, 30]]);
    /// 
    /// let z = x + y;
    /// 
    /// assert_eq!(&[[0, 11], [22, 33]], z.get_values());
    /// ```
    fn add(self, rhs: Matrix<TR, R, C>) -> Self::Output {
        let values: [[TO; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| self[r][c] + rhs[r][c]).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[TO; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<TL, TR, const R: usize, const C: usize> AddAssign<Matrix<TR, R, C>> for Matrix<TL, R, C>
where
    TL: Copy,
    TL: Add<TR, Output = TL>,
    TR: Copy,
{
    /// Normal elementwise addition of two matrices
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = static_linear_algebra::Matrix::new(&[[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::Matrix::new(&[[0, 10], [20, 30]]);
    /// 
    /// x += y;
    /// 
    /// assert_eq!(&[[0, 11], [22, 33]], x.get_values());
    /// ```
    fn add_assign(&mut self, rhs: Matrix<TR, R, C>) {
        let values: [[TL; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| self[r][c] + rhs[r][c]).collect::<Vec<TL>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[TL; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        self.values = values;
    }
}

impl<TL, TR, TO, const R: usize, const C: usize> Sub<Matrix<TR, R, C>> for Matrix<TL, R, C>
where
    TL: Copy,
    TL: Sub<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
{
    type Output = Matrix<TO, R, C>;

    /// Normal elementwise subtraction of two matrices
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new(&[[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::Matrix::new(&[[0, 10], [20, 30]]);
    /// 
    /// let z = x - y;
    /// 
    /// assert_eq!(&[[0, -9], [-18, -27]], z.get_values());
    /// ```
    fn sub(self, rhs: Matrix<TR, R, C>) -> Self::Output {
        let values: [[TO; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| self[r][c] - rhs[r][c]).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[TO; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<TL, TR, const R: usize, const C: usize> SubAssign<Matrix<TR, R, C>> for Matrix<TL, R, C>
where
    TL: Copy,
    TL: Sub<TR, Output = TL>,
    TR: Copy,
{
    /// Normal elementwise subtraction of two matrices
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = static_linear_algebra::Matrix::new(&[[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::Matrix::new(&[[0, 10], [20, 30]]);
    /// 
    /// x -= y;
    /// 
    /// assert_eq!(&[[0, -9], [-18, -27]], x.get_values());
    /// ```
    fn sub_assign(&mut self, rhs: Matrix<TR, R, C>) {
        let values: [[TL; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| self[r][c] - rhs[r][c]).collect::<Vec<TL>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[TL; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        self.values = values;
    }
}

impl<TL, TR, TO, const R: usize, const K: usize, const C: usize> Mul<Matrix<TR, K, C>> for Matrix<TL, R, K>
where
    TL: Copy,
    TL: Mul<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
    TO: Sum,
{
    type Output = Matrix<TO, R, C>;

    /// Normal matrix multiplication
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new(&[[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::Matrix::new(&[[0, 10], [20, 30]]);
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(&[[20, 30], [60, 110]], z.get_values());
    /// ```
    fn mul(self, rhs: Matrix<TR, K, C>) -> Self::Output {
        let values: [[TO; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| 
            (0..K).map(|k| self[r][k] * rhs[k][c]).sum()).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[TO; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<TL, TR, TO, const R: usize, const C: usize> Mul<VectorColumn<TR, C>> for Matrix<TL, R, C>
where
    TL: Copy,
    TL: Mul<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
    TO: Sum,
{
    type Output = VectorColumn<TO, R>;

    /// Right hand side vector multiplication
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new(&[[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::VectorColumn::new(&[0, 10]);
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(&[10, 30], z.get_values());
    /// ```
    fn mul(self, rhs: VectorColumn<TR, C>) -> Self::Output {
        let values: [TO; R] = match (0..R).map(|r| (0..C).map(|c| self[r][c] * rhs[c]).sum()).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<TL, TR, const S: usize> MulAssign<Matrix<TR, S, S>> for Matrix<TL, S, S>
where
    TL: Copy,
    TL: Mul<TR, Output = TL>,
    TL: Sum,
    TR: Copy,
{
    /// Normal matrix multiplication
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = static_linear_algebra::Matrix::new(&[[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::Matrix::new(&[[0, 10], [20, 30]]);
    /// 
    /// x *= y;
    /// 
    /// assert_eq!(&[[20, 30], [60, 110]], x.get_values());
    /// ```
    fn mul_assign(&mut self, rhs: Matrix<TR, S, S>) {
        let values: [[TL; S]; S] = 
            match (0..S).map(|r| 
            match (0..S).map(|c| 
            (0..S).map(|k| self[r][k] * rhs[k][c]).sum()).collect::<Vec<TL>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[TL; S]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        self.values = values;
    }
}

impl<TL, TR, TO, const R: usize, const C: usize> Mul<TR> for Matrix<TL, R, C>
where
    TL: Copy,
    TL: Mul<TR, Output = TO>,
    TR: Copy,
    TR: Num,
    TO: Copy,
{
    type Output = Matrix<TO, R, C>;

    /// Scalar multiplication from the right, this is preferable from lhs scalar multiplication
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new(&[[0, 1], [2, 3]]);
    /// let y = 10;
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(&[[0, 10], [20, 30]], z.get_values());
    /// ```
    fn mul(self, rhs: TR) -> Self::Output {
        let values: [[TO; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| self[r][c] * rhs).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[TO; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<TL, TR, const R: usize, const C: usize> MulAssign<TR> for Matrix<TL, R, C>
where
    TL: Copy,
    TL: Mul<TR, Output = TL>,
    TR: Copy,
    TR: Num,
{
    /// Scalar multiplication from the right, this is preferable from lhs scalar multiplication
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = static_linear_algebra::Matrix::new(&[[0, 1], [2, 3]]);
    /// let y = 10;
    /// 
    /// x *= y;
    /// 
    /// assert_eq!(&[[0, 10], [20, 30]], x.get_values());
    /// ```
    fn mul_assign(&mut self, rhs: TR) {
        let values: [[TL; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| self[r][c] * rhs).collect::<Vec<TL>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[TL; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        self.values = values;
    }
}

macro_rules! dot_method {
    ($TL:ty) => {
        impl<TR, TO, const R: usize, const C: usize> Mul<Matrix<TR, R, C>> for $TL
        where
            $TL: Mul<TR, Output = TO>,
            TR: Copy,
            TO: Copy,
        {
            type Output = Matrix<TO, R, C>;

            /// Scalar multiplication from the left, this only works for specific types, for generic types use rhs multiplication
            fn mul(self, rhs: Matrix<TR, R, C>) -> Self::Output {
                let values: [[TO; C]; R] = 
                    match (0..R).map(|r| 
                    match (0..C).map(|c| self * rhs[r][c]).collect::<Vec<TO>>().try_into() {
                    Ok(result) => result,
                    Err(_) => panic!("Should not happen"),
                }).collect::<Vec<[TO; C]>>().try_into() {
                    Ok(result) => result,
                    Err(_) => panic!("Should not happen"),
                };

                Self::Output {values}
            }
        }
    };
}

impl<T, TR, TO, const R: usize, const C: usize> Mul<Matrix<TR, R, C>> for Complex<T>
where
    Complex<T>: Copy,
    Complex<T>: Mul<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
{
    type Output = Matrix<TO, R, C>;

    /// Scalar multiplication from the left, this only works for specific types, for generic types use rhs multiplication
    fn mul(self, rhs: Matrix<TR, R, C>) -> Self::Output {
        let values: [[TO; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| self * rhs[r][c]).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[TO; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

dot_method!(u8);
dot_method!(u16);
dot_method!(u32);
dot_method!(u64);
dot_method!(u128);
dot_method!(usize);
dot_method!(i8);
dot_method!(i16);
dot_method!(i32);
dot_method!(i64);
dot_method!(i128);
dot_method!(isize);
dot_method!(f32);
dot_method!(f64);

#[cfg(test)]
mod tests {
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
    fn add_assign() {
        let mut a = Matrix::new(&[[0, 1], [2, 3]]);
        let b = Matrix::new(&[[0, 10], [20, 30]]);
        a += b;
        assert_eq!([[0, 11], [22, 33]], a.values);
    }

    #[test]
    fn sub() {
        let a = Matrix::new(&[[0, 1], [2, 3]]);
        let b = Matrix::new(&[[0, 10], [20, 30]]);
        let c = a - b;
        assert_eq!([[0, -9], [-18, -27]], c.values);
    }

    #[test]
    fn sub_assign() {
        let mut a = Matrix::new(&[[0, 1], [2, 3]]);
        let b = Matrix::new(&[[0, 10], [20, 30]]);
        a -= b;
        assert_eq!([[0, -9], [-18, -27]], a.values);
    }

    #[test]
    fn mul() {
        let matrix1 = Matrix::new(&[[0, 1, 2, 3], [4, 5, 6, 7]]);
        let matrix2 = Matrix::new(&[[0, 10, 20], [30, 40, 50], [60, 70, 80], [90, 100, 110]]);
        let result = matrix1 * matrix2;
        assert_eq!([[420, 480, 540], [1140, 1360, 1580]], result.values);    
    }

    #[test]
    fn mul_assign() {
        let mut matrix1 = Matrix::new(&[[0, 1], [2, 3]]);
        let matrix2 = Matrix::new(&[[0, 10], [20, 30]]);
        matrix1 *= matrix2;
        assert_eq!([[20, 30], [60, 110]], matrix1.values);    
    }

    #[test]
    fn mul_vector() {
        let matrix = Matrix::new(&[[0, 1, 2], [3, 4, 5]]);
        let vector = VectorColumn::new(&[0, 1, 2]);
        let result = matrix * vector;
        assert_eq!([5, 14], result.values);
    }

    #[test]
    fn scalar_mul_right() {
        let matrix = Matrix::new(&[[0, 1, 2]]);
        let result = matrix * 4;
        assert_eq!([[0, 4, 8]], result.values);    
    }

    #[test]
    fn scalar_mul_assign() {
        let mut matrix = Matrix::new(&[[0, 1, 2]]);
        matrix *= 4;
        assert_eq!([[0, 4, 8]], matrix.values);    
    }

    macro_rules! dot_method_test {
        ($T:ty, $name:ident) => {
            #[test]
            fn $name() {
                let matrix: Matrix<$T, 1, 3> = Matrix::new(&[[0 as $T, 1 as $T, 2 as $T]]);
                let result = (4 as $T) * matrix;
                assert_eq!([[0 as $T, 4 as $T, 8 as $T]], result.values);    
            }
        };
    }

    dot_method_test!(u8, scalar_mul_left_u8);
    dot_method_test!(u16, scalar_mul_left_u16);
    dot_method_test!(u32, scalar_mul_left_u32);
    dot_method_test!(u64, scalar_mul_left_u64);
    dot_method_test!(u128, scalar_mul_left_u128);
    dot_method_test!(usize, scalar_mul_left_usize);
    dot_method_test!(i8, scalar_mul_left_i8);
    dot_method_test!(i16, scalar_mul_left_i16);
    dot_method_test!(i32, scalar_mul_left_i32);
    dot_method_test!(i64, scalar_mul_left_i64);
    dot_method_test!(i128, scalar_mul_left_i128);
    dot_method_test!(isize, scalar_mul_left_isize);
    dot_method_test!(f32, scalar_mul_left_f32);
    dot_method_test!(f64, scalar_mul_left_f64);

    #[test]
    fn scalar_mul_left_complex() {
        let matrix = Matrix::new(&[[Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 1.)]]);
        let result = Complex::new(0., 4.) * matrix;
        assert_eq!([[Complex::new(0., 0.), Complex::new(0., 4.), Complex::new(-4., 0.)]], result.values);    
    }

    #[test]
    fn transpose() {
        let matrix = Matrix::new(&[[0, 1, 2], [3, 4, 5]]);
        let result = matrix.transpose();
        assert_eq!([[0, 3], [1, 4], [2, 5]], result.values);
    }

    #[test]
    fn hermitian_conjugate() {
        let matrix = Matrix::new(&[[Complex::new(0, 0), Complex::new(1, 0), Complex::new(0, 1)], [Complex::new(1, 1), Complex::new(-1, 0), Complex::new(0, -1)]]);
        let result = matrix.transpose();
        assert_eq!([[Complex::new(0, 0), Complex::new(1, 1)], [Complex::new(1, 0), Complex::new(-1, 0)], [Complex::new(0, 1), Complex::new(0, -1)]], result.values);
    }

    #[test]
    fn is_hermitian() {
        let matrix1 = Matrix::new(&[[Complex::new(1, 0), Complex::new(0, 1)], [Complex::new(0, -1), Complex::new(0, 0)]]);
        let matrix2 = Matrix::new(&[[Complex::new(1, 0), Complex::new(0, 1)], [Complex::new(0, -1), Complex::new(0, 1)]]);
        assert_eq!(true, matrix1.is_hermitian());
        assert_eq!(false, matrix2.is_hermitian());
    }

    #[test]
    fn is_symmetric() {
        let matrix1 = Matrix::new(&[[0, 1], [1, 2]]);
        let matrix2 = Matrix::new(&[[0, 1], [2, 1]]);
        assert_eq!(true, matrix1.is_symmetric());
        assert_eq!(false, matrix2.is_symmetric());
    }
}
