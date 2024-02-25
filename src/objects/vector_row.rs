//! Implementation and all methods on row vectors

use std::{
    ops::{
        Index,
        IndexMut,
        Add,
        Sub,
        AddAssign,
        SubAssign,
        Mul,
        MulAssign,
    },
    iter::Sum,
};
use num::{
    traits::{
        Zero,
        Num,
    },
    Complex,
};
use core::ops::Neg;
use super::{
    Matrix,
    VectorColumn,
};

/// A static row vector type
/// 
/// Size must be known at compile time but operations are checked for size compatibility at compile time too
/// 
/// S: The length of the vector
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct VectorRow<T, const S: usize>
where
    T: Copy,
{
    pub(crate) values: [T; S],
}

impl<T, const S: usize> VectorRow<T, S>
where
    T: Copy,
{
    /// Initializes a new row vector with the given values
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::VectorRow::new([0, 1, 2]);
    /// 
    /// assert_eq!(&[0, 1, 2], x.get_values());
    /// ```
    pub fn new(values: [T; S]) -> Self {
        assert_ne!(S, 0);
        
        Self {
            values
        }
    }

    /// Initializes a new row vector filled with a single value
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::VectorRow::<f32, 2>::from_value(1.);
    /// 
    /// assert_eq!(&[1., 1.], x.get_values());
    /// ```
    pub fn from_value(value: T) -> Self {
        assert_ne!(S, 0);

        Self {
            values: [value; S]
        }
    }

    /// Retrieves a reference to the data of the row vector
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::VectorRow::new([0, 1]);
    /// let data = x.get_values();
    /// 
    /// assert_eq!(&[0, 1], data);
    /// ```
    pub fn get_values(&self) -> &[T; S] {
        &self.values
    }

    /// Retrieves a mutable reference to the data of the row vector
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = static_linear_algebra::VectorRow::new([0, 1]);
    /// let data = x.get_values_mut();
    /// data[0] = 5;
    /// 
    /// assert_eq!(&[5, 1], x.get_values());
    /// ```
    pub fn get_values_mut(&mut self) -> &mut [T; S] {
        &mut self.values
    }

    /// Transposes the row vector into a column vector
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::VectorRow::new([0, 1, 2]);
    /// let y = x.transpose();
    /// 
    /// assert_eq!(&[0, 1, 2], y.get_values());
    /// ```
    pub fn transpose(&self) -> VectorColumn<T, S> {
        VectorColumn {values: self.values}
    }
}

impl<T, const S: usize> VectorRow<Complex<T>, S> 
where
    T: Copy,
    T: Num,
    T: Neg<Output = T>,
{
    /// Takes the hermitian conjugate of the row vector (transpose the vector 
    /// and complex conjugate each element (change the sign of the imaginary part))
    /// 
    /// # Examples
    /// 
    /// ```
    /// use num::Complex;
    /// 
    /// let x = static_linear_algebra::VectorRow::new([Complex::new(1, 0), Complex::new(0, 2)]);
    /// let y = x.hermitian_conjugate();
    /// 
    /// assert_eq!(&[Complex::new(1, 0), Complex::new(0, -2)], y.get_values())
    /// ```
    pub fn hermitian_conjugate(&self) -> VectorColumn<Complex<T>, S> {
        let values: [Complex<T>; S] = match (0..S).map(|i| self.values[i].conj()).collect::<Vec<Complex<T>>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        VectorColumn {values}
    }
}

impl<T, const S: usize> Index<usize> for VectorRow<T, S> 
where
    T: Copy,
{
    type Output = T;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.values[idx]
    }
}

impl<T, const S: usize> IndexMut<usize> for VectorRow<T, S> 
where
    T: Copy,
{
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.values[idx]
    }
}

impl<T, const S: usize> Zero for VectorRow<T, S>
where
    T: Copy,
    T: Zero,
    T: PartialEq,
{
    fn zero() -> Self {
        Self::from_value(T::zero())
    }

    fn is_zero(&self) -> bool {
        (0..S).all(|i| self.values[i] == T::zero())
    }
}

impl<T, const S: usize> Sum for VectorRow<T, S>
where
    T: Copy,
    T: Zero,
    T: Add<T, Output = T>,
    T: PartialEq,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut result: Self = VectorRow::zero();

        for value in iter {
            result += value;
        }

        result
    }
}

impl<'a, T, const S: usize> Sum<&'a VectorRow<T, S>> for VectorRow<T, S>
where
    T: Copy,
    T: Zero,
    T: Add<T, Output = T>,
    T: PartialEq,
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let mut result: Self = VectorRow::zero();

        for value in iter {
            result += *value;
        }

        result
    }
}

impl<TL, TR, TO, const S: usize> Add<VectorRow<TR, S>> for VectorRow<TL, S>
where
    TL: Copy,
    TL: Add<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
{
    type Output = VectorRow<TO, S>;

    /// Normal elementwise addition of two row vectors
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::VectorRow::new([0, 1]);
    /// let y = static_linear_algebra::VectorRow::new([0, 10]);
    /// 
    /// let z = x + y;
    /// 
    /// assert_eq!(&[0, 11], z.get_values());
    /// ```
    fn add(self, rhs: VectorRow<TR, S>) -> Self::Output {
        let values: [TO; S] = match (0..S).map(|i| self.values[i] + rhs.values[i]).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<T, const S: usize> AddAssign<VectorRow<T, S>> for VectorRow<T, S>
where
    T: Copy,
    T: Add<T, Output = T>,
{
    /// Normal elementwise addition of two row vectors
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = static_linear_algebra::VectorRow::new([0, 1]);
    /// let y = static_linear_algebra::VectorRow::new([0, 10]);
    /// 
    /// x += y;
    /// 
    /// assert_eq!(&[0, 11], x.get_values());
    /// ```
    fn add_assign(&mut self, rhs: VectorRow<T, S>) {
        let values: [T; S] = match (0..S).map(|i| self.values[i] + rhs.values[i]).collect::<Vec<T>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        self.values = values;
    }
}

impl<TL, TR, TO, const S: usize> Sub<VectorRow<TR, S>> for VectorRow<TL, S>
where
    TL: Copy,
    TL: Sub<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
{
    type Output = VectorRow<TO, S>;

    /// Normal elementwise subtraction of two row vectors
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::VectorRow::new([0, 1]);
    /// let y = static_linear_algebra::VectorRow::new([0, 10]);
    /// 
    /// let z = x - y;
    /// 
    /// assert_eq!(&[0, -9], z.get_values());
    /// ```
    fn sub(self, rhs: VectorRow<TR, S>) -> Self::Output {
        let values: [TO; S] = match (0..S).map(|i| self.values[i] - rhs.values[i]).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<T, const S: usize> SubAssign<VectorRow<T, S>> for VectorRow<T, S>
where
    T: Copy,
    T: Sub<T, Output = T>,
{
    /// Normal elementwise subtraction of two row vectors
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = static_linear_algebra::VectorRow::new([0, 1]);
    /// let y = static_linear_algebra::VectorRow::new([0, 10]);
    /// 
    /// x -= y;
    /// 
    /// assert_eq!(&[0, -9], x.get_values());
    /// ```
    fn sub_assign(&mut self, rhs: VectorRow<T, S>) {
        let values: [T; S] = match (0..S).map(|i| self.values[i] - rhs.values[i]).collect::<Vec<T>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        self.values = values;
    }
}

impl<TL, TR, TO, const S: usize> Mul<VectorRow<TR, S>> for VectorRow<TL, S>
where
    TL: Copy,
    TL: Mul<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
    TO: Sum,
{
    type Output = TO;

    /// Inner product (dot product) between two row vectors
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::VectorRow::new([0, 1]);
    /// let y = static_linear_algebra::VectorRow::new([0, 10]);
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(10, z);
    /// ```
    fn mul(self, rhs: VectorRow<TR, S>) -> Self::Output {
        (0..S).map(|i| self.values[i] * rhs.values[i]).sum()
    }
}

impl<TL, TR, TO, const S: usize> Mul<VectorColumn<TR, S>> for VectorRow<TL, S>
where
    TL: Copy,
    TL: Mul<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
    TO: Sum,
{
    type Output = TO;

    /// Inner product between a row vector and a column vector
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::VectorRow::new([0, 1]);
    /// let y = static_linear_algebra::VectorColumn::new([10, 20]);
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(20, z);
    /// ```
    fn mul(self, rhs: VectorColumn<TR, S>) -> Self::Output {
        (0..S).map(|i| self.values[i] * rhs.values[i]).sum()
    }
}

impl<TL, TR, TO, const R: usize, const C: usize> Mul<Matrix<TR, R, C>> for VectorRow<TL, R>
where
    TL: Copy,
    TL: Mul<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
    TO: Sum,
{
    type Output = VectorRow<TO, C>;

    /// Multiplication between a row vector and a matrix
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::VectorRow::new([0, 1]);
    /// let y = static_linear_algebra::Matrix::new([[0, 10, 20], [30, 40, 50]]);
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(&[30, 40, 50], z.get_values());
    /// ```
    fn mul(self, rhs: Matrix<TR, R, C>) -> Self::Output {
        let values: [TO; C] = match (0..C).map(|c| (0..R).map(|r| self.values[r] * rhs.values[r][c]).sum()).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<TL, TR, TO, const S: usize> Mul<TR> for VectorRow<TL, S>
where
    TL: Copy,
    TL: Mul<TR, Output = TO>,
    TR: Copy,
    TR: Num,
    TO: Copy,
{
    type Output = VectorRow<TO, S>;

    /// Scalar multiplication from the right, this is preferable from lhs scalar multiplication
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = static_linear_algebra::VectorRow::new([0, 1]);
    /// let y = 10;
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(&[0, 10], z.get_values());
    /// ```
    fn mul(self, rhs: TR) -> Self::Output {
        let values: [TO; S] = match (0..S).map(|i| self.values[i] * rhs).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<T, const S: usize> MulAssign<T> for VectorRow<T, S>
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
    /// let mut x = static_linear_algebra::VectorRow::new([0, 1]);
    /// let y = 10;
    /// 
    /// x *= y;
    /// 
    /// assert_eq!(&[0, 10], x.get_values());
    /// ```
    fn mul_assign(&mut self, rhs: T) {
        let values: [T; S] = match (0..S).map(|i| self.values[i] * rhs).collect::<Vec<T>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        self.values = values;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let result = VectorRow::new([0, 1, 2, 3]);
        assert_eq!([0, 1, 2, 3], result.values);
    }

    #[test]
    fn from_value() {
        let result: VectorRow<f64, 5> = VectorRow::from_value(5.6);
        assert_eq!([5.6, 5.6, 5.6, 5.6, 5.6], result.values);
    }

    #[test]
    fn get_values() {
        let result = VectorRow::new([0, 1, 2]);
        assert_eq!([0, 1, 2], *result.get_values());
    }

    #[test]
    fn get_values_mut() {
        let mut result = VectorRow::new([0, 1, 2]);
        let data = result.get_values_mut();
        data[1] = 5;
        assert_eq!([0, 5, 2], result.values);
    }

    #[test]
    fn index() {
        let result = VectorRow::new([0, 1, 2]);
        assert_eq!(1, result[1]);
    }

    #[test]
    fn index_mut() {
        let mut result = VectorRow::new([0, 1, 2]);
        result[1] = 5;
        assert_eq!([0, 5, 2], result.values);
    }

    #[test]
    fn zero() {
        let result: VectorRow<i32, 5> = VectorRow::zero();
        assert_eq!([0, 0, 0, 0, 0], result.values);
    }

    #[test]
    fn is_zero() {
        let result1 = VectorRow::new([0, 0, 0]);
        let result2 = VectorRow::new([0, 1, 0]);
        assert_eq!(true, result1.is_zero());
        assert_eq!(false, result2.is_zero());
    }

    #[test]
    fn sum() {
        let list: [VectorRow<i32, 3>; 3] = [VectorRow::new([0, 1, 2]), VectorRow::new([0, 10, 20]), VectorRow::new([0, 100, 200])];
        let result: VectorRow<i32, 3> = list.into_iter().sum();
        assert_eq!([0, 111, 222], result.values);
    }

    #[test]
    fn sum_ref() {
        let list: [VectorRow<i32, 3>; 3] = [VectorRow::new([0, 1, 2]), VectorRow::new([0, 10, 20]), VectorRow::new([0, 100, 200])];
        let result: VectorRow<i32, 3> = list.iter().sum();
        assert_eq!([0, 111, 222], result.values);
    }

    #[test]
    fn add() {
        let vector1 = VectorRow::new([0, 1, 2]);
        let vector2 = VectorRow::new([0, 10, 20]);
        let result = vector1 + vector2;
        assert_eq!([0, 11, 22], result.values);
    }

    #[test]
    fn add_assign() {
        let mut vector1 = VectorRow::new([0, 1, 2]);
        let vector2 = VectorRow::new([0, 10, 20]);
        vector1 += vector2;
        assert_eq!([0, 11, 22], vector1.values);
    }

    #[test]
    fn sub() {
        let vector1 = VectorRow::new([0, 1, 2]);
        let vector2 = VectorRow::new([0, 10, 20]);
        let result = vector1 - vector2;
        assert_eq!([0, -9, -18], result.values);
    }

    #[test]
    fn sub_assign() {
        let mut vector1 = VectorRow::new([0, 1, 2]);
        let vector2 = VectorRow::new([0, 10, 20]);
        vector1 -= vector2;
        assert_eq!([0, -9, -18], vector1.values);
    }

    #[test]
    fn dot_product() {
        let vector1 = VectorRow::new([0, 1, 2]);
        let vector2 = VectorRow::new([3, 4, 5]);
        let result = vector1 * vector2;
        assert_eq!(14, result);
    }

    #[test]
    fn dot_product_column() {
        let vector1 = VectorRow::new([0, 1, 2]);
        let vector2 = VectorColumn::new([3, 4, 5]);
        let result = vector1 * vector2;
        assert_eq!(14, result);
    }

    #[test]
    fn mul_matrix() {
        let matrix = Matrix::new([[0, 1, 2], [3, 4, 5]]);
        let vector = VectorRow::new([0, 1]);
        let result = vector * matrix;
        assert_eq!([3, 4, 5], result.values);
    }

    #[test]
    fn scalar_mul() {
        let vector = VectorRow::new([0, 1, 2]);
        let result = vector * 5;
        assert_eq!([0, 5, 10], result.values);
    }

    #[test]
    fn scalar_mul_assign() {
        let mut vector = VectorRow::new([0, 1, 2]);
        vector *= 5;
        assert_eq!([0, 5, 10], vector.values);
    }

    #[test]
    fn transpose() {
        let vector = VectorRow::new([0, 1, 2]);
        let result = vector.transpose();
        assert_eq!([0, 1, 2], result.values);
    }

    #[test]
    fn hermitian_conjugate() {
        let vector = VectorRow::new([Complex::new(0, 0), Complex::new(1, 0), Complex::new(0, 1)]);
        let result = vector.hermitian_conjugate();
        assert_eq!([Complex::new(0, 0), Complex::new(1, 0), Complex::new(0, -1)], result.values);
    }
}
