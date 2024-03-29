//! Implementation of and all methods on matrices

use std::{
    ops::{
        Index,
        IndexMut,
        Add,
        Sub,
        Mul,
        Div,
        AddAssign,
        SubAssign,
        MulAssign,
    },
    iter::{
        Sum,
        Product,
    },
    array,
};
use num::{
    traits::{
        Zero,
        One,
        Num,
    },
    Complex,
};
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
    /// let x = static_linear_algebra::Matrix::new([[0, 1, 2], [3, 4, 5]]);
    /// 
    /// assert_eq!(&[[0, 1, 2], [3, 4, 5]], x.get_values());
    /// ```
    pub fn new(values: [[T; C]; R]) -> Self {
        assert_ne!(C, 0);
        assert_ne!(R, 0);

        Self {
            values
        }
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
        assert_ne!(C, 0);
        assert_ne!(R, 0);

        Self {
            values: [[value; C]; R]
        }
    }

    /// Retrieves a reference to the data of the matrix
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new([[0, 1], [2, 3]]);
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
    /// let mut x = static_linear_algebra::Matrix::new([[0, 1], [2, 3]]);
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
    /// let x = static_linear_algebra::Matrix::new([[0, 1, 2], [3, 4, 5]]);
    /// let y = x.transpose();
    /// 
    /// assert_eq!(&[[0, 3], [1, 4], [2, 5]], y.get_values());
    /// ```
    pub fn transpose(&self) -> Matrix<T, C, R> {
        let values: [[T; R]; C] = 
            match (0..C).map(|r| 
            match (0..R).map(|c| self.values[c][r]).collect::<Vec<T>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[T; R]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Matrix {
            values
        }
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
        assert_ne!(S, 0);

        let mut use_values= [[T::zero(); S]; S];
        
        for (n, value) in values.iter().enumerate() {
            use_values[n][n] = *value;
        }

        Self {
            values: use_values
        }
    }
}

impl<T, const S: usize> Matrix<T, S, S>
where
    T: Copy,
    T: Zero,
    T: One,
{
    /// Initializes a diagonal matrix where the diagonal contains ones
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::<i32, 3, 3>::identity();
    /// 
    /// assert_eq!(&[[1, 0, 0], [0, 1, 0], [0, 0, 1]], x.get_values());
    /// ```
    pub fn identity() -> Self {
        assert_ne!(S, 0);

        let mut use_values= [[T::zero(); S]; S];
        
        for n in 0..S {
            use_values[n][n] = T::one();
        }

        Self {
            values: use_values
        }
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
    /// let x = static_linear_algebra::Matrix::new([[Complex::new(1, 0), Complex::new(0, 2)], [Complex::new(0, 3), Complex::new(0, 4)]]);
    /// let y = x.hermitian_conjugate();
    /// 
    /// assert_eq!(&[[Complex::new(1, 0), Complex::new(0, -3)], [Complex::new(0, -2), Complex::new(0, -4)]], y.get_values())
    /// ```
    pub fn hermitian_conjugate(&self) -> Matrix<Complex<T>, C, R> {
        let values: [[Complex<T>; R]; C] = 
            match (0..C).map(|r| 
            match (0..R).map(|c| self.values[c][r].conj()).collect::<Vec<Complex<T>>>().try_into() {
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
    /// let x = static_linear_algebra::Matrix::new([[0, 1], [1, 2]]);
    /// 
    /// assert_eq!(true, x.is_symmetric());
    /// ```
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new([[0, 1], [2, 1]]);
    /// 
    /// assert_eq!(false, x.is_symmetric());
    /// ```
    pub fn is_symmetric(&self) -> bool {
        (0..S)
            .all(|r| {
                (0..r + 1)
                    .all(|c| self.values[r][c] == self.values[c][r])
            })
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
    /// let x = static_linear_algebra::Matrix::new([[Complex::new(0, 0), Complex::new(0, 1)], [Complex::new(0, -1), Complex::new(2, 0)]]);
    /// 
    /// assert_eq!(true, x.is_hermitian());
    /// ```
    /// 
    /// ```
    /// use num::Complex;
    /// 
    /// let x = static_linear_algebra::Matrix::new([[Complex::new(0, 0), Complex::new(0, 1)], [Complex::new(0, 1), Complex::new(2, 0)]]);
    /// 
    /// assert_eq!(false, x.is_hermitian());
    /// ```
    pub fn is_hermitian(&self) -> bool {
        (0..S)
            .all(|r| {
                (0..r + 1).all(|c| self.values[r][c] == self.values[c][r].conj())
            })
    }
}

impl<T, const S: usize> Matrix<T, S, S> 
where
    T: Copy,
    T: Mul<T, Output = T>,
    T: Sum,
    T: Add<Output = T>,
    T: Sub<Output = T>,
    T: Neg<Output = T>,
{
    /// Calculates the determinant for the matrix
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// 
    /// assert_eq!(x.determinant(), -2.0)
    /// ```
    pub fn determinant(&self) -> T {
        match S {
            1 => {
                self.values[0][0]
            }
            2 => {
                self.values[0][0] * self.values[1][1] - self.values[0][1] * self.values[1][0]
            }
            3 => {
                self.values[0][0] * (self.values[1][1] * self.values[2][2] - self.values[1][2] * self.values[2][1]) -
                self.values[0][1] * (self.values[1][0] * self.values[2][2] - self.values[1][2] * self.values[2][0]) +
                self.values[0][2] * (self.values[1][0] * self.values[2][1] - self.values[1][1] * self.values[2][0])
            }
            4 => {
                self.values[0][0] * (
                    self.values[1][1] * (self.values[2][2] * self.values[3][3] - self.values[2][3] * self.values[3][2]) -
                    self.values[1][2] * (self.values[2][1] * self.values[3][3] - self.values[2][3] * self.values[3][1]) +
                    self.values[1][3] * (self.values[2][1] * self.values[3][2] - self.values[2][2] * self.values[3][1])
                ) -
                self.values[0][1] * (
                    self.values[1][0] * (self.values[2][2] * self.values[3][3] - self.values[2][3] * self.values[3][2]) -
                    self.values[1][2] * (self.values[2][0] * self.values[3][3] - self.values[2][3] * self.values[3][0]) +
                    self.values[1][3] * (self.values[2][0] * self.values[3][2] - self.values[2][2] * self.values[3][0])
                ) +
                self.values[0][2] * (
                    self.values[1][0] * (self.values[2][1] * self.values[3][3] - self.values[2][3] * self.values[3][1]) -
                    self.values[1][1] * (self.values[2][0] * self.values[3][3] - self.values[2][3] * self.values[3][0]) +
                    self.values[1][3] * (self.values[2][0] * self.values[3][1] - self.values[2][1] * self.values[3][0])
                ) -
                self.values[0][3] * (
                    self.values[1][0] * (self.values[2][1] * self.values[3][2] - self.values[2][2] * self.values[3][1]) -
                    self.values[1][1] * (self.values[2][0] * self.values[3][2] - self.values[2][2] * self.values[3][0]) +
                    self.values[1][2] * (self.values[2][0] * self.values[3][1] - self.values[2][1] * self.values[3][0])
                )
            }
            _ => determinant_step(&self.values, &[true; S]),
        }
    }
}

fn determinant_step<T, const S: usize>(data: &[[T; S]], unused: &[bool; S]) -> T
where
    T: Copy,
    T: Mul<T, Output = T>,
    T: Sum,
    T: Neg<Output = T>,
{
    // Run through the first row and multiply unused values by subdeterminants of the next rows
    data[0]
        .iter()
        .enumerate()
        .zip(unused.iter())
        .filter_map(|(value, &keep)| {
            if keep {
                Some(value)
            } else {
                None
            }
        })
        .enumerate()
        .map(|(sign, (location, &value))| {
            if data.len() <= 1 {
                // Just return the value
                value
            } else {
                // Remove the current column from the unused columns list
                let new_unused: [bool; S] = unused
                    .iter()
                    .enumerate()
                    .map(|(keep_location, &keep)| {
                        if keep_location == location {
                            false
                        } else {
                            keep
                        }
                    })
                    .collect::<Vec<bool>>()
                    .try_into()
                    .unwrap();

                // Calculate the sub determinant
                let sub_det_value = determinant_step(&data[1..], &new_unused);

                // Make sure the sign alternates
                if sign % 2 == 0 {
                    value * sub_det_value
                } else {
                    -value * sub_det_value
                }
            }
        })
        .sum::<T>()
}

impl<T, const S: usize> Matrix<T, S, S> 
where
    T: Copy,
    T: Mul<T, Output = T>,
    T: Div<T, Output = T>,
    T: Sum,
    T: Add<Output = T>,
    T: Sub<Output = T>,
    T: Neg<Output = T>,
    T: Zero,
    T: One,
{
    /// Calculates the inverse for the matrix
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// 
    /// assert_eq!(x.inverse().unwrap().get_values(), &[[-2.0, 1.0], [1.5, -0.5]])
    /// ```
    pub fn inverse(&self) -> Option<Self> {
        // Get the determinant and make sure it is not zero
        let det = self.determinant();
        if det.is_zero() {
            return None;
        }

        // Calculate the inverse
        let values: [[T; S]; S] = array::from_fn(|row| {
            let unused: [bool; S] = array::from_fn(|unused_column| unused_column != row);
            array::from_fn(|column| {
                let value = sub_determinant_step(&self.values, &unused, column) / det;
                if (row + column) % 2 == 0 {
                    value
                } else {
                    -value
                }
            })
        });

        Some(Self {
            values
        })
    }
}

fn sub_determinant_step<T, const S: usize>(data: &[[T; S]], unused: &[bool; S], skip_line: usize) -> T
where
    T: Copy,
    T: Mul<T, Output = T>,
    T: Sum,
    T: Neg<Output = T>,
    T: One,
{
    // Stop if done
    if data.is_empty() {
        return T::one();
    }

    // Try to skip line
    if skip_line == 0 {
        return sub_determinant_step(&data[1..], unused, data.len());
    }

    // Run through the first row and multiply unused values by subdeterminants of the next rows
    data[0]
        .iter()
        .enumerate()
        .zip(unused.iter())
        .filter_map(|(value, &keep)| {
            if keep {
                Some(value)
            } else {
                None
            }
        })
        .enumerate()
        .map(|(sign, (location, &value))| {
            // Remove the current column from the unused columns list
            let new_unused: [bool; S] = unused
                .iter()
                .enumerate()
                .map(|(keep_location, &keep)| {
                    if keep_location == location {
                        false
                    } else {
                        keep
                    }
                })
                .collect::<Vec<bool>>()
                .try_into()
                .unwrap();

            // Calculate the sub determinant
            let sub_det_value = sub_determinant_step(&data[1..], &new_unused, skip_line - 1);

            // Make sure the sign alternates
            if sign % 2 == 0 {
                value * sub_det_value
            } else {
                -value * sub_det_value
            }
        })
        .sum::<T>()
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
        self.values
            .iter()
            .all(|column| {
                column
                    .iter()
                    .all(|value| *value == T::zero())
            })
    }
}

impl<T, const C: usize> One for Matrix<T, C, C>
where
    T: Copy,
    T: Zero,
    T: One,
    T: Mul,
    T: Sum,
{
    fn one() -> Self {
        Self::identity()
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
            result += value;
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
            result += *value;
        }

        result
    }
}

impl<T, const C: usize> Product for Matrix<T, C, C>
where
    T: Copy,
    T: Zero,
    T: One,
    T: Mul<T, Output = T>,
    T: Sum,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut result: Self = Matrix::one();

        for value in iter {
            result *= value;
        }

        result
    }
}

impl<'a, T, const C: usize> Product<&'a Matrix<T, C, C>> for Matrix<T, C, C>
where
    T: Copy,
    T: Zero,
    T: One,
    T: Mul<T, Output = T>,
    T: Sum,
{
    fn product<I: Iterator<Item = &'a Matrix<T, C, C>>>(iter: I) -> Self {
        let mut result: Self = Matrix::one();

        for value in iter {
            result *= *value;
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
    /// let x = static_linear_algebra::Matrix::new([[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::Matrix::new([[0, 10], [20, 30]]);
    /// 
    /// let z = x + y;
    /// 
    /// assert_eq!(&[[0, 11], [22, 33]], z.get_values());
    /// ```
    fn add(self, rhs: Matrix<TR, R, C>) -> Self::Output {
        let values: [[TO; C]; R] = match self.values
            .iter()
            .zip(rhs.values.iter())
            .map(|(c_lhs, c_rhs)| {
                match c_lhs
                    .iter()
                    .zip(c_rhs.iter())
                    .map(|(lhs, rhs)| *lhs + *rhs)
                    .collect::<Vec<TO>>()
                    .try_into() {
                        Ok(result) => result,
                        Err(_) => panic!("Should not happen"),
                    }
            })
            .collect::<Vec<[TO; C]>>()
            .try_into() {
                Ok(result) => result,
                Err(_) => panic!("Should not happen"),
            };

        Self::Output {
            values
        }
    }
}

impl<T, const R: usize, const C: usize> AddAssign<Matrix<T, R, C>> for Matrix<T, R, C>
where
    T: Copy,
    T: Add<T, Output = T>,
{
    /// Normal elementwise addition of two matrices
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = static_linear_algebra::Matrix::new([[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::Matrix::new([[0, 10], [20, 30]]);
    /// 
    /// x += y;
    /// 
    /// assert_eq!(&[[0, 11], [22, 33]], x.get_values());
    /// ```
    fn add_assign(&mut self, rhs: Matrix<T, R, C>) {
        let result = *self + rhs;
        self.values = result.values;
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
    /// let x = static_linear_algebra::Matrix::new([[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::Matrix::new([[0, 10], [20, 30]]);
    /// 
    /// let z = x - y;
    /// 
    /// assert_eq!(&[[0, -9], [-18, -27]], z.get_values());
    /// ```
    fn sub(self, rhs: Matrix<TR, R, C>) -> Self::Output {
        let values: [[TO; C]; R] = match self.values
            .iter()
            .zip(rhs.values.iter())
            .map(|(c_lhs, c_rhs)| {
                match c_lhs
                    .iter()
                    .zip(c_rhs.iter())
                    .map(|(lhs, rhs)| *lhs - *rhs)
                    .collect::<Vec<TO>>()
                    .try_into() {
                        Ok(result) => result,
                        Err(_) => panic!("Should not happen"),
                    }
            })
            .collect::<Vec<[TO; C]>>()
            .try_into() {
                Ok(result) => result,
                Err(_) => panic!("Should not happen"),
            };

        Self::Output {
            values
        }
    }
}

impl<T, const R: usize, const C: usize> SubAssign<Matrix<T, R, C>> for Matrix<T, R, C>
where
    T: Copy,
    T: Sub<T, Output = T>,
{
    /// Normal elementwise subtraction of two matrices
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = static_linear_algebra::Matrix::new([[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::Matrix::new([[0, 10], [20, 30]]);
    /// 
    /// x -= y;
    /// 
    /// assert_eq!(&[[0, -9], [-18, -27]], x.get_values());
    /// ```
    fn sub_assign(&mut self, rhs: Matrix<T, R, C>) {
        let result = *self - rhs;
        self.values = result.values;
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
    /// let x = static_linear_algebra::Matrix::new([[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::Matrix::new([[0, 10], [20, 30]]);
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(&[[20, 30], [60, 110]], z.get_values());
    /// ```
    fn mul(self, rhs: Matrix<TR, K, C>) -> Self::Output {
        let values: [[TO; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| 
            (0..K).map(|k| self.values[r][k] * rhs.values[k][c]).sum()).collect::<Vec<TO>>().try_into() {
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
    /// let x = static_linear_algebra::Matrix::new([[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::VectorColumn::new([0, 10]);
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(&[10, 30], z.get_values());
    /// ```
    fn mul(self, rhs: VectorColumn<TR, C>) -> Self::Output {
        let values: [TO; R] = match (0..R).map(|r| (0..C).map(|c| self.values[r][c] * rhs.values[c]).sum()).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
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
    /// let x = static_linear_algebra::Matrix::new([[0, 1], [2, 3]]);
    /// let y = 10;
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(&[[0, 10], [20, 30]], z.get_values());
    /// ```
    fn mul(self, rhs: TR) -> Self::Output {
        let values: [[TO; C]; R] = match self.values
            .iter()
            .map(|column| {
                match column
                    .iter()
                    .map(|value| *value * rhs)
                    .collect::<Vec<TO>>()
                    .try_into() {
                        Ok(result) => result,
                        Err(_) => panic!("Should not happen"),
                    }
            })
            .collect::<Vec<[TO; C]>>()
            .try_into() {
                Ok(result) => result,
                Err(_) => panic!("Should not happen"),
            };

        Self::Output {
            values
        }
    }
}

impl<T, const S: usize> MulAssign<Matrix<T, S, S>> for Matrix<T, S, S>
where
    T: Copy,
    T: Mul<T, Output = T>,
    T: Sum,
{
    /// Normal matrix multiplication
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = static_linear_algebra::Matrix::new([[0, 1], [2, 3]]);
    /// let y = static_linear_algebra::Matrix::new([[0, 10], [20, 30]]);
    /// 
    /// x *= y;
    /// 
    /// assert_eq!(&[[20, 30], [60, 110]], x.get_values());
    /// ```
    fn mul_assign(&mut self, rhs: Matrix<T, S, S>) {
        let result = *self * rhs;
        self.values = result.values;
    }
}

impl<T, const R: usize, const C: usize> MulAssign<T> for Matrix<T, R, C>
where
    T: Copy,
    T: Mul<T, Output = T>,
    T: Num,
{
    /// Scalar multiplication from the right, this is preferable from lhs scalar multiplication
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = static_linear_algebra::Matrix::new([[0, 1], [2, 3]]);
    /// let y = 10;
    /// 
    /// x *= y;
    /// 
    /// assert_eq!(&[[0, 10], [20, 30]], x.get_values());
    /// ```
    fn mul_assign(&mut self, rhs: T) {
        let result = *self * rhs;
        self.values = result.values;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_close<const R: usize, const C: usize>(lhs: &[[f64; C]; R], rhs: &[[f64; C]; R], tolerance: f64) -> bool {
        lhs.iter()
            .zip(rhs.iter())
            .all(|(lhs, rhs)| {
                lhs.iter()
                    .zip(rhs.iter())
                    .all(|(&lhs, &rhs)| {
                        (lhs - rhs).abs() < tolerance
                    })
            })
    }

    #[test]
    fn new() {
        let result = Matrix::new([[0, 1], [2, 3], [4, 5]]);
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
    fn identity() {
        let result = Matrix::<i32, 3, 3>::identity();
        assert_eq!([[1, 0, 0], [0, 1, 0], [0, 0, 1]], result.values);
    }

    #[test]
    fn get_values() {
        let result = Matrix::new([[0, 1, 2], [3, 4, 5]]);
        assert_eq!(&[[0, 1, 2], [3, 4, 5]], result.get_values());
    }

    #[test]
    fn get_values_mut() {
        let mut result = Matrix::new([[0, 1, 2], [3, 4, 5]]);
        let data = result.get_values_mut();
        data[0][1] = 10;
        assert_eq!([[0, 10, 2], [3, 4, 5]], result.values);
    }

    #[test]
    fn matrix_of_matrix() {
        let result = Matrix::new([[Matrix::new([[1, 0], [0, 1]])], [Matrix::new([[0, 1], [-1, 0]])]]);
        assert_eq!([[Matrix::new([[1, 0], [0, 1]])], [Matrix::new([[0, 1], [-1, 0]])]], result.values);
    }

    #[test]
    fn index_get() {
        let result = Matrix::new([[0, 1, 2], [3, 4, 5]]);
        assert_eq!(2, result[0][2]);
    }

    #[test]
    fn index_set() {
        let mut result = Matrix::new([[0, 1, 2], [3, 4, 5]]);
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
        let list: [Matrix<i32, 2, 2>; 3] = [Matrix::new([[0, 1], [2, 3]]), Matrix::new([[0, 10], [20, 30]]), Matrix::new([[0, 100], [200, 300]])];
        let result: Matrix<i32, 2, 2> = list.into_iter().sum();
        assert_eq!([[0, 111], [222, 333]], result.values);
    }

    #[test]
    fn sum_ref() {
        let list: [Matrix<i32, 2, 2>; 3] = [Matrix::new([[0, 1], [2, 3]]), Matrix::new([[0, 10], [20, 30]]), Matrix::new([[0, 100], [200, 300]])];
        let result: Matrix<i32, 2, 2> = list.into_iter().sum();
        assert_eq!([[0, 111], [222, 333]], result.values);
    }

    #[test]
    fn product() {
        let list: [Matrix<i32, 2, 2>; 3] = [Matrix::new([[0, 1], [2, 3]]), Matrix::new([[4, 5], [6, 7]]), Matrix::new([[8, 9], [10, 11]])];
        let result: Matrix<i32, 2, 2> = list.into_iter().product();
        assert_eq!([[118, 131], [518, 575]], result.values);
    }

    #[test]
    fn product_ref() {
        let list: [Matrix<i32, 2, 2>; 3] = [Matrix::new([[0, 1], [2, 3]]), Matrix::new([[4, 5], [6, 7]]), Matrix::new([[8, 9], [10, 11]])];
        let result: Matrix<i32, 2, 2> = list.iter().product();
        assert_eq!([[118, 131], [518, 575]], result.values);
    }

    #[test]
    fn add() {
        let a = Matrix::new([[0, 1], [2, 3]]);
        let b = Matrix::new([[0, 10], [20, 30]]);
        let c = a + b;
        assert_eq!([[0, 11], [22, 33]], c.values);
    }

    #[test]
    fn add_assign() {
        let mut a = Matrix::new([[0, 1], [2, 3]]);
        let b = Matrix::new([[0, 10], [20, 30]]);
        a += b;
        assert_eq!([[0, 11], [22, 33]], a.values);
    }

    #[test]
    fn sub() {
        let a = Matrix::new([[0, 1], [2, 3]]);
        let b = Matrix::new([[0, 10], [20, 30]]);
        let c = a - b;
        assert_eq!([[0, -9], [-18, -27]], c.values);
    }

    #[test]
    fn sub_assign() {
        let mut a = Matrix::new([[0, 1], [2, 3]]);
        let b = Matrix::new([[0, 10], [20, 30]]);
        a -= b;
        assert_eq!([[0, -9], [-18, -27]], a.values);
    }

    #[test]
    fn mul() {
        let matrix1 = Matrix::new([[0, 1, 2, 3], [4, 5, 6, 7]]);
        let matrix2 = Matrix::new([[0, 10, 20], [30, 40, 50], [60, 70, 80], [90, 100, 110]]);
        let result = matrix1 * matrix2;
        assert_eq!([[420, 480, 540], [1140, 1360, 1580]], result.values);    
    }

    #[test]
    fn mul_assign() {
        let mut matrix1 = Matrix::new([[0, 1], [2, 3]]);
        let matrix2 = Matrix::new([[0, 10], [20, 30]]);
        matrix1 *= matrix2;
        assert_eq!([[20, 30], [60, 110]], matrix1.values);    
    }

    #[test]
    fn mul_vector() {
        let matrix = Matrix::new([[0, 1, 2], [3, 4, 5]]);
        let vector = VectorColumn::new([0, 1, 2]);
        let result = matrix * vector;
        assert_eq!([5, 14], result.values);
    }

    #[test]
    fn scalar_mul() {
        let matrix = Matrix::new([[0, 1, 2]]);
        let result = matrix * 4;
        assert_eq!([[0, 4, 8]], result.values);    
    }

    #[test]
    fn scalar_mul_assign() {
        let mut matrix = Matrix::new([[0, 1, 2]]);
        matrix *= 4;
        assert_eq!([[0, 4, 8]], matrix.values);    
    }

    #[test]
    fn transpose() {
        let matrix = Matrix::new([[0, 1, 2], [3, 4, 5]]);
        let result = matrix.transpose();
        assert_eq!([[0, 3], [1, 4], [2, 5]], result.values);
    }

    #[test]
    fn hermitian_conjugate() {
        let matrix = Matrix::new([[Complex::new(0, 0), Complex::new(1, 0), Complex::new(0, 1)], [Complex::new(1, 1), Complex::new(-1, 0), Complex::new(0, -1)]]);
        let result = matrix.transpose();
        assert_eq!([[Complex::new(0, 0), Complex::new(1, 1)], [Complex::new(1, 0), Complex::new(-1, 0)], [Complex::new(0, 1), Complex::new(0, -1)]], result.values);
    }

    #[test]
    fn is_hermitian() {
        let matrix1 = Matrix::new([[Complex::new(1, 0), Complex::new(0, 1)], [Complex::new(0, -1), Complex::new(0, 0)]]);
        let matrix2 = Matrix::new([[Complex::new(1, 0), Complex::new(0, 1)], [Complex::new(0, -1), Complex::new(0, 1)]]);
        assert_eq!(true, matrix1.is_hermitian());
        assert_eq!(false, matrix2.is_hermitian());
    }

    #[test]
    fn is_symmetric() {
        let matrix1 = Matrix::new([[0, 1], [1, 2]]);
        let matrix2 = Matrix::new([[0, 1], [2, 1]]);
        assert_eq!(true, matrix1.is_symmetric());
        assert_eq!(false, matrix2.is_symmetric());
    }

    #[test]
    fn determinant() {
        let matrix1 = Matrix::new([[5]]);
        let matrix2 = Matrix::new([[4, 5], [7, 4]]);
        let matrix3 = Matrix::new([[3, 9, 4], [6, 3, 6], [1, 6, 1]]);
        let matrix4 = Matrix::new([[3, 5, 8, 3], [8, 8, 4, 4], [9, 3, 1, 6], [7, 3, 8, 6]]);
        let matrix5 = Matrix::new([[9, 2, 7, 5, 6], [0, 0, 9, 3, 6], [0, 9, 8, 7, 8], [3, 9, 1, 5, 9], [1, 2, 2, 2, 7]]);
        let matrix6 = Matrix::new([[9, 0, 8, 0, 1, 0], [5, 2, 3, 1, 2, 3], [4, 1, 1, 1, 6, 3], [5, 2, 8, 8, 8, 2], [8, 8, 6, 8, 9, 6], [8, 3, 7, 2, 2, 5]]);
        assert_eq!(5, matrix1.determinant());
        assert_eq!(-19, matrix2.determinant());
        assert_eq!(33, matrix3.determinant());
        assert_eq!(272, matrix4.determinant());
        assert_eq!(1569, matrix5.determinant());
        assert_eq!(-3458, matrix6.determinant());
    }

    #[test]
    fn inverse() {
        let matrix1 = Matrix::new([[5.0]]);
        let matrix2 = Matrix::new([[4.0, 5.0], [7.0, 4.0]]);
        let matrix3 = Matrix::new([[3.0, 9.0, 4.0], [6.0, 3.0, 6.0], [1.0, 6.0, 1.0]]);
        let matrix4 = Matrix::new([[3.0, 5.0, 8.0, 3.0], [8.0, 8.0, 4.0, 4.0], [9.0, 3.0, 1.0, 6.0], [7.0, 3.0, 8.0, 6.0]]);
        let matrix5 = Matrix::new([[9.0, 2.0, 7.0, 5.0, 6.0], [0.0, 0.0, 9.0, 3.0, 6.0], [0.0, 9.0, 8.0, 7.0, 8.0], [3.0, 9.0, 1.0, 5.0, 9.0], [1.0, 2.0, 2.0, 2.0, 7.0]]);
        let matrix6 = Matrix::new([[9.0, 0.0, 8.0, 0.0, 1.0, 0.0], [5.0, 2.0, 3.0, 1.0, 2.0, 3.0], [4.0, 1.0, 1.0, 1.0, 6.0, 3.0], [5.0, 2.0, 8.0, 8.0, 8.0, 2.0], [8.0, 8.0, 6.0, 8.0, 9.0, 6.0], [8.0, 3.0, 7.0, 2.0, 2.0, 5.0]]);
        assert!(check_close(matrix1.inverse().unwrap().get_values(), &[[0.2]], 1e-5));
        assert!(check_close(matrix2.inverse().unwrap().get_values(), &[[-0.210526, 0.263158], [0.368421, -0.210526]], 1e-5));
        assert!(check_close(matrix3.inverse().unwrap().get_values(), &[[-1.0, 0.454545, 1.27273], [0.0, -0.030303, 0.181818], [1.0, -0.272727, -1.36364]], 1e-5));
        assert!(check_close(matrix4.inverse().unwrap().get_values(), &[[-0.926471, 0.540441, -0.632353, 0.735294], [0.455882, -0.0992647, 0.279412, -0.441176], [-0.264706, 0.154412, -0.323529, 0.352941], [1.20588, -0.786765, 1.02941, -0.941176]], 1e-5));
        assert!(check_close(matrix5.inverse().unwrap().get_values(), &[[0.0210325, 0.335883, -0.328872, 0.432122, -0.48566], [-0.225621, 0.912046, -0.65392, 1.09178, -1.24474], [-0.137667, 0.771192, -0.483748, 0.717017, -0.912046], [0.493308, -2.15233, 1.6501, -2.31931, 2.51816], [-0.040153, 0.0860421, -0.0994264, 0.08413, 0.108987]], 1e-5));
        assert!(check_close(matrix6.inverse().unwrap().get_values(), &[[-0.0237131, 1.81029, -0.32273, 0.175824, -0.172932, -0.75535], [0.187392, -1.50087, 0.1845, -0.340659, 0.390977, 0.456912], [0.137941, -1.89647, 0.316368, -0.181319, 0.176692, 0.80856], [-0.224407, 2.20474, -0.541932, 0.395604, -0.270677, -0.831116], [0.10989, -1.12088, 0.373626, -0.131868, 0.142857, 0.32967], [-0.221805, 0.225564, 0.0300752, 0.0714286, -0.154135, 0.203008]], 1e-5));

        let matrix_singular = Matrix::new([[2.0, 3.0], [4.0, 6.0]]);
        assert_eq!(matrix_singular.inverse(), None);
    }
}
