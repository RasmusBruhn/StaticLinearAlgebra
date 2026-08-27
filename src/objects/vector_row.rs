//! Implementation and all methods on row vectors

use super::VectorColumn;
use itertools::Itertools;
use num::{
    traits::{Num, Zero},
    Complex,
};
use std::{
    convert::{From, Into},
    fmt::Debug,
    iter::Sum,
    ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// A static row vector type
///
/// Size must be known at compile time but operations are checked for size compatibility at compile time too
///
/// S: The length of the vector
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct VectorRow<T, const S: usize>
where
    T: Copy + Debug + PartialEq,
{
    pub(crate) values: [T; S],
}

impl<T, const S: usize> VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
{
    /// Retrieves a reference to the data of the row vector
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::from([0, 1]);
    /// let data = x.get_values();
    ///
    /// assert_eq!(&[0, 1], data);
    /// ```
    pub fn get_values(&self) -> &[T; S] {
        return &self.values;
    }

    /// Retrieves a mutable reference to the data of the row vector
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = static_linear_algebra::VectorRow::from([0, 1]);
    /// let data = x.get_values_mut();
    /// data[0] = 5;
    ///
    /// assert_eq!(&[5, 1], x.get_values());
    /// ```
    pub fn get_values_mut(&mut self) -> &mut [T; S] {
        return &mut self.values;
    }

    /// Runs the given function on each element of the row vector and returns
    /// a new row vector with the results
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::from([1, 2, 3]);
    /// let y = x.elementwise(|v| v * 2);
    ///
    /// assert_eq!(&[2, 4, 6], y.get_values());
    /// ```
    pub fn elementwise<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        let values = self
            .values
            .iter()
            .map(|&value| f(value))
            .collect_array()
            .expect("Should not happen");

        return Self { values };
    }

    /// Runs the given function on each element of the row vector in place
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = static_linear_algebra::VectorRow::from([1, 2, 3]);
    /// x.elementwise_mut(|v| v * 2);
    ///
    /// assert_eq!(&[2, 4, 6], x.get_values());
    /// ```
    pub fn elementwise_mut<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        for value in &mut self.values {
            *value = f(*value);
        }
    }

    /// Runs the given operation on each pair of elements from two column
    /// vectors and returns a new row vector with the results
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::from([2, 20]);
    /// let y = static_linear_algebra::VectorRow::from([1, 10]);
    /// let z = x.elementwise_operation(&y, |a, b| a + b);
    ///
    /// assert_eq!(&[3, 30], z.get_values());
    /// ```
    pub fn elementwise_operation<F>(&self, rhs: &Self, f: F) -> Self
    where
        F: Fn(T, T) -> T,
    {
        let values = self
            .values
            .iter()
            .zip(rhs.get_values().iter())
            .map(|(&a, &b)| f(a, b))
            .collect_array()
            .expect("Should not happen");

        return Self { values };
    }

    /// Runs the given operation on each pair of elements from two column
    /// vectors in place of the first vector
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = static_linear_algebra::VectorRow::from([2, 20]);
    /// let y = static_linear_algebra::VectorRow::from([1, 10]);
    /// x.elementwise_operation_mut(&y, |a, b| a + b);
    ///
    /// assert_eq!(&[3, 30], x.get_values());
    /// ```
    pub fn elementwise_operation_mut<F>(&mut self, rhs: &Self, f: F)
    where
        F: Fn(T, T) -> T,
    {
        for (a, &b) in self.values.iter_mut().zip(rhs.get_values().iter()) {
            *a = f(*a, b);
        }
    }

    /// Transposes the row vector into a column vector
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::from([0, 1, 2]);
    /// let y = x.transpose();
    ///
    /// assert_eq!(&[0, 1, 2], y.get_values());
    /// ```
    pub fn transpose(&self) -> VectorColumn<T, S> {
        return VectorColumn {
            values: self.values,
        };
    }
}

impl<T, const S: usize> VectorRow<Complex<T>, S>
where
    T: Copy + Debug + PartialEq,
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
    /// let x = static_linear_algebra::VectorRow::from([Complex::new(1, 0), Complex::new(0, 2)]);
    /// let y = x.hermitian_conjugate();
    ///
    /// assert_eq!(&[Complex::new(1, 0), Complex::new(0, -2)], y.get_values())
    /// ```
    pub fn hermitian_conjugate(&self) -> VectorColumn<Complex<T>, S> {
        let values = self
            .values
            .iter()
            .map(|value| value.conj())
            .collect_array()
            .expect("Should not happen");

        return VectorColumn { values };
    }
}

impl<T, const S: usize> From<[T; S]> for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
{
    /// Constructs a row vector from an array of values
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::from([1, 2, 3]);
    ///
    /// assert_eq!(&[1, 2, 3], x.get_values());
    /// ```
    fn from(values: [T; S]) -> Self {
        return Self { values };
    }
}

impl<T, const S: usize> From<T> for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
{
    /// Constructs a row vector with all elements equal to the given value
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, 3>::from(5);
    ///
    /// assert_eq!(&[5, 5, 5], x.get_values());
    /// ```
    fn from(value: T) -> Self {
        return Self { values: [value; S] };
    }
}

impl<T, const S: usize> Into<[T; S]> for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
{
    /// Converts the row vector into an array of values
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::from([1, 2, 3]);
    /// let arr: [i32; 3] = x.into();
    ///
    /// assert_eq!([1, 2, 3], arr);
    /// ```
    fn into(self) -> [T; S] {
        return self.values;
    }
}

impl<T, const S: usize> Index<usize> for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
{
    type Output = T;

    /// Accesses an element of the row vector by index
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::from([0, 1, 2]);
    ///
    /// assert_eq!(0, x[0]);
    /// assert_eq!(1, x[1]);
    /// assert_eq!(2, x[2]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        return &self.values[idx];
    }
}

impl<T, const S: usize> IndexMut<usize> for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
{
    /// Mutable access of an element of the row vector by index
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = static_linear_algebra::VectorRow::from([0, 1, 2]);
    ///
    /// x[0] = 10;
    /// x[1] = 20;
    /// x[2] = 30;
    ///
    /// assert_eq!(10, x[0]);
    /// assert_eq!(20, x[1]);
    /// assert_eq!(30, x[2]);
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        return &mut self.values[idx];
    }
}

impl<T, const S: usize> Zero for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
    T: Zero,
{
    /// Constructs a row vector with all elements equal to the zero element
    ///
    /// # Examples
    ///
    /// ```
    /// use num::traits::identities::Zero;
    ///
    /// let x = static_linear_algebra::VectorRow::<i32, 3>::zero();
    ///
    /// assert_eq!(&[0, 0, 0], x.get_values());
    /// ```
    fn zero() -> Self {
        return Self::from(T::zero());
    }

    /// Checks if the row vector is a zero vector (all elements are zero)
    ///
    /// # Examples
    ///
    /// ```
    /// use num::traits::identities::Zero;
    ///
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([0, 0, 0]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 0, 0]);
    ///
    /// assert!(x.is_zero());
    /// assert!(!y.is_zero());
    /// ```
    fn is_zero(&self) -> bool {
        return self.values.iter().all(|&i| i == T::zero());
    }
}

impl<T, const S: usize> Sum for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
    T: Zero,
    T: Add<T, Output = T>,
{
    /// Performs an elementwise sum over all row vectors in an iterator
    ///
    /// # Examples
    ///
    /// ```
    /// let vectors = [
    ///     static_linear_algebra::VectorRow::from([1, 10, 100]),
    ///     static_linear_algebra::VectorRow::from([2, 20, 200]),
    ///     static_linear_algebra::VectorRow::from([3, 30, 300]),
    /// ];
    ///
    /// let result = vectors.into_iter().sum::<static_linear_algebra::VectorRow::<i32, _>>();
    ///
    /// assert_eq!(&[6, 60, 600], result.get_values());
    /// ```
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut result: Self = VectorRow::zero();

        for value in iter {
            result = result + value;
        }

        return result;
    }
}

impl<'a, T, const S: usize> Sum<&'a VectorRow<T, S>> for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
    T: Zero,
    T: Add<T, Output = T>,
{
    /// Performs an elementwise sum over all row vectors in an iterator
    ///
    /// # Examples
    ///
    /// ```
    /// let vectors = [
    ///     static_linear_algebra::VectorRow::from([1, 10, 100]),
    ///     static_linear_algebra::VectorRow::from([2, 20, 200]),
    ///     static_linear_algebra::VectorRow::from([3, 30, 300]),
    /// ];
    ///
    /// let result = vectors.iter().sum::<static_linear_algebra::VectorRow::<i32, _>>();
    ///
    /// assert_eq!(&[6, 60, 600], result.get_values());
    /// ```
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let mut result: Self = VectorRow::zero();

        for value in iter {
            result = result + value;
        }

        return result;
    }
}

impl<TL, TR, TO, const S: usize> Add<&VectorRow<TR, S>> for &VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Add<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TO: Copy + Debug + PartialEq,
{
    type Output = VectorRow<TO, S>;

    /// Normal elementwise addition of two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([2, 20]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    ///
    /// let z = &x + &y;
    ///
    /// assert_eq!(&[3, 30], z.get_values());
    /// ```
    fn add(self, rhs: &VectorRow<TR, S>) -> Self::Output {
        let values = self
            .values
            .iter()
            .zip(rhs.values.iter())
            .map(|(&a, &b)| a + b)
            .collect_array()
            .expect("Should not happen");

        return Self::Output { values };
    }
}

impl<TL, TR, TO, const S: usize> Add<&VectorRow<TR, S>> for VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Add<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TO: Copy + Debug + PartialEq,
{
    type Output = VectorRow<TO, S>;

    /// Normal elementwise addition of two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([2, 20]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    ///
    /// let z = x + &y;
    ///
    /// assert_eq!(&[3, 30], z.get_values());
    /// ```
    fn add(self, rhs: &VectorRow<TR, S>) -> Self::Output {
        return &self + rhs;
    }
}

impl<TL, TR, TO, const S: usize> Add<VectorRow<TR, S>> for &VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Add<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TO: Copy + Debug + PartialEq,
{
    type Output = VectorRow<TO, S>;

    /// Normal elementwise addition of two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([2, 20]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    ///
    /// let z = &x + y;
    ///
    /// assert_eq!(&[3, 30], z.get_values());
    /// ```
    fn add(self, rhs: VectorRow<TR, S>) -> Self::Output {
        return self + &rhs;
    }
}

impl<TL, TR, TO, const S: usize> Add<VectorRow<TR, S>> for VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Add<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TO: Copy + Debug + PartialEq,
{
    type Output = VectorRow<TO, S>;

    /// Normal elementwise addition of two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([2, 20]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    ///
    /// let z = x + y;
    ///
    /// assert_eq!(&[3, 30], z.get_values());
    /// ```
    fn add(self, rhs: VectorRow<TR, S>) -> Self::Output {
        return &self + &rhs;
    }
}

impl<T, const S: usize> AddAssign<&VectorRow<T, S>> for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
    T: Add<T, Output = T>,
{
    /// Normal in place elementwise addition of two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = static_linear_algebra::VectorRow::from([2, 20]);
    /// let y = static_linear_algebra::VectorRow::from([1, 10]);
    ///
    /// x += &y;
    ///
    /// assert_eq!(&[3, 30], x.get_values());
    /// ```
    fn add_assign(&mut self, rhs: &VectorRow<T, S>) {
        self.values
            .iter_mut()
            .zip(rhs.values.iter())
            .for_each(|(a, &b)| *a = *a + b);
    }
}

impl<T, const S: usize> AddAssign<VectorRow<T, S>> for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
    T: Add<T, Output = T>,
{
    /// Normal in place elementwise addition of two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = static_linear_algebra::VectorRow::from([2, 20]);
    /// let y = static_linear_algebra::VectorRow::from([1, 10]);
    ///
    /// x += y;
    ///
    /// assert_eq!(&[3, 30], x.get_values());
    /// ```
    fn add_assign(&mut self, rhs: VectorRow<T, S>) {
        *self += &rhs;
    }
}

impl<TL, TR, TO, const S: usize> Sub<&VectorRow<TR, S>> for &VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Sub<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TO: Copy + Debug + PartialEq,
{
    type Output = VectorRow<TO, S>;

    /// Normal elementwise subtraction of two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([3, 30]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    ///
    /// let z = &x - &y;
    ///
    /// assert_eq!(&[2, 20], z.get_values());
    /// ```
    fn sub(self, rhs: &VectorRow<TR, S>) -> Self::Output {
        let values = self
            .values
            .iter()
            .zip(rhs.values.iter())
            .map(|(&a, &b)| a - b)
            .collect_array()
            .expect("Should not happen");

        return Self::Output { values };
    }
}

impl<TL, TR, TO, const S: usize> Sub<&VectorRow<TR, S>> for VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Sub<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TO: Copy + Debug + PartialEq,
{
    type Output = VectorRow<TO, S>;

    /// Normal elementwise subtraction of two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([3, 30]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    ///
    /// let z = x - &y;
    ///
    /// assert_eq!(&[2, 20], z.get_values());
    /// ```
    fn sub(self, rhs: &VectorRow<TR, S>) -> Self::Output {
        return &self - rhs;
    }
}

impl<TL, TR, TO, const S: usize> Sub<VectorRow<TR, S>> for &VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Sub<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TO: Copy + Debug + PartialEq,
{
    type Output = VectorRow<TO, S>;

    /// Normal elementwise subtraction of two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([3, 30]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    ///
    /// let z = &x - y;
    ///
    /// assert_eq!(&[2, 20], z.get_values());
    /// ```
    fn sub(self, rhs: VectorRow<TR, S>) -> Self::Output {
        return self - &rhs;
    }
}

impl<TL, TR, TO, const S: usize> Sub<VectorRow<TR, S>> for VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Sub<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TO: Copy + Debug + PartialEq,
{
    type Output = VectorRow<TO, S>;

    /// Normal elementwise subtraction of two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([3, 30]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    ///
    /// let z = x - y;
    ///
    /// assert_eq!(&[2, 20], z.get_values());
    /// ```
    fn sub(self, rhs: VectorRow<TR, S>) -> Self::Output {
        return &self - &rhs;
    }
}

impl<T, const S: usize> SubAssign<&VectorRow<T, S>> for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
    T: Sub<T, Output = T>,
{
    /// Normal in place elementwise subtraction of two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = static_linear_algebra::VectorRow::from([3, 30]);
    /// let y = static_linear_algebra::VectorRow::from([1, 10]);
    ///
    /// x -= &y;
    ///
    /// assert_eq!(&[2, 20], x.get_values());
    /// ```
    fn sub_assign(&mut self, rhs: &VectorRow<T, S>) {
        self.values
            .iter_mut()
            .zip(rhs.values.iter())
            .for_each(|(a, &b)| *a = *a - b);
    }
}

impl<T, const S: usize> SubAssign<VectorRow<T, S>> for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
    T: Sub<T, Output = T>,
{
    /// Normal in place elementwise subtraction of two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = static_linear_algebra::VectorRow::from([3, 30]);
    /// let y = static_linear_algebra::VectorRow::from([1, 10]);
    ///
    /// x -= y;
    ///
    /// assert_eq!(&[2, 20], x.get_values());
    /// ```
    fn sub_assign(&mut self, rhs: VectorRow<T, S>) {
        *self -= &rhs;
    }
}

impl<TI, TO, const S: usize> Neg for &VectorRow<TI, S>
where
    TI: Copy + Debug + PartialEq,
    TI: Neg<Output = TO>,
    TO: Copy + Debug + PartialEq,
{
    type Output = VectorRow<TO, S>;

    /// Elementwise negation of the row vector
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([3, 30]);
    /// let y = -&x;
    ///
    /// assert_eq!(&[-3, -30], y.get_values());
    /// ```
    fn neg(self) -> Self::Output {
        let values = self.values.map(|a| -a);
        return VectorRow { values };
    }
}

impl<TI, TO, const S: usize> Neg for VectorRow<TI, S>
where
    TI: Copy + Debug + PartialEq,
    TI: Neg<Output = TO>,
    TO: Copy + Debug + PartialEq,
{
    type Output = VectorRow<TO, S>;

    /// Elementwise negation of the row vector
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([3, 30]);
    /// let y = -x;
    ///
    /// assert_eq!(&[-3, -30], y.get_values());
    /// ```
    fn neg(self) -> Self::Output {
        return -&self;
    }
}

impl<TL, TR, TO, const S: usize> Mul<&VectorRow<TR, S>> for &VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Mul<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TO: Copy + Debug + PartialEq,
    TO: Sum,
{
    type Output = TO;

    /// Inner product (dot product) between two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([2, 20]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    ///
    /// let z = &x * &y;
    ///
    /// assert_eq!(202, z);
    /// ```
    fn mul(self, rhs: &VectorRow<TR, S>) -> Self::Output {
        return self
            .values
            .iter()
            .zip(rhs.values.iter())
            .map(|(&a, &b)| a * b)
            .sum();
    }
}

impl<TL, TR, TO, const S: usize> Mul<&VectorRow<TR, S>> for VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Mul<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TO: Copy + Debug + PartialEq,
    TO: Sum,
{
    type Output = TO;

    /// Inner product (dot product) between two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([2, 20]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    ///
    /// let z = x * &y;
    ///
    /// assert_eq!(202, z);
    /// ```
    fn mul(self, rhs: &VectorRow<TR, S>) -> Self::Output {
        return &self * rhs;
    }
}

impl<TL, TR, TO, const S: usize> Mul<VectorRow<TR, S>> for &VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Mul<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TO: Copy + Debug + PartialEq,
    TO: Sum,
{
    type Output = TO;

    /// Inner product (dot product) between two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([2, 20]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    ///
    /// let z = &x * y;
    ///
    /// assert_eq!(202, z);
    /// ```
    fn mul(self, rhs: VectorRow<TR, S>) -> Self::Output {
        return self * &rhs;
    }
}

impl<TL, TR, TO, const S: usize> Mul<VectorRow<TR, S>> for VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Mul<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TO: Copy + Debug + PartialEq,
    TO: Sum,
{
    type Output = TO;

    /// Inner product (dot product) between two row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([2, 20]);
    /// let y = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    ///
    /// let z = x * y;
    ///
    /// assert_eq!(202, z);
    /// ```
    fn mul(self, rhs: VectorRow<TR, S>) -> Self::Output {
        return &self * &rhs;
    }
}

impl<TL, TR, TO, const S: usize> Mul<TR> for &VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Mul<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TR: Num,
    TO: Copy + Debug + PartialEq,
{
    type Output = VectorRow<TO, S>;

    /// Scalar multiplication from the right
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    /// let y = 2;
    ///
    /// let z = &x * y;
    ///
    /// assert_eq!(&[2, 20], z.get_values());
    /// ```
    fn mul(self, rhs: TR) -> Self::Output {
        let values = self
            .values
            .iter()
            .map(|&v| v * rhs)
            .collect_array()
            .expect("Should not happen");

        return Self::Output { values };
    }
}

impl<TL, TR, TO, const S: usize> Mul<TR> for VectorRow<TL, S>
where
    TL: Copy + Debug + PartialEq,
    TL: Mul<TR, Output = TO>,
    TR: Copy + Debug + PartialEq,
    TR: Num,
    TO: Copy + Debug + PartialEq,
{
    type Output = VectorRow<TO, S>;

    /// Scalar multiplication from the right
    ///
    /// # Examples
    ///
    /// ```
    /// let x = static_linear_algebra::VectorRow::<i32, _>::from([1, 10]);
    /// let y = 2;
    ///
    /// let z = x * y;
    ///
    /// assert_eq!(&[2, 20], z.get_values());
    /// ```
    fn mul(self, rhs: TR) -> Self::Output {
        return &self * rhs;
    }
}

impl<T, const S: usize> MulAssign<T> for VectorRow<T, S>
where
    T: Copy + Debug + PartialEq,
    T: Mul<T, Output = T>,
    T: Num,
{
    /// Scalar multiplication from the right, this is preferable from lhs scalar multiplication
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = static_linear_algebra::VectorRow::from([1, 10]);
    /// let y = 2;
    ///
    /// x *= y;
    ///
    /// assert_eq!(&[2, 20], x.get_values());
    /// ```
    fn mul_assign(&mut self, rhs: T) {
        self.values.iter_mut().for_each(|v| *v = *v * rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_array() {
        let result = VectorRow::from([0, 1, 2, 3]);
        assert_eq!([0, 1, 2, 3], result.values);
    }

    #[test]
    fn from_value() {
        let result: VectorRow<f64, 5> = VectorRow::from(5.6);
        assert_eq!([5.6, 5.6, 5.6, 5.6, 5.6], result.values);
    }

    #[test]
    fn into_array() {
        let input = VectorRow::from([0, 1, 2, 3]);
        let result: [i32; _] = input.into();
        assert_eq!([0, 1, 2, 3], result);
    }

    #[test]
    fn get_values() {
        let result = VectorRow::from([0, 1, 2]);
        assert_eq!([0, 1, 2], *result.get_values());
    }

    #[test]
    fn get_values_mut() {
        let mut result = VectorRow::from([0, 1, 2]);
        let data = result.get_values_mut();
        data[1] = 5;
        assert_eq!([0, 5, 2], result.values);
    }

    #[test]
    fn index() {
        let result = VectorRow::from([0, 1, 2]);
        assert_eq!(1, result[1]);
    }

    #[test]
    fn index_mut() {
        let mut result = VectorRow::from([0, 1, 2]);
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
        let result1 = VectorRow::<i32, _>::from([0, 0, 0]);
        let result2 = VectorRow::<i32, _>::from([0, 1, 0]);
        assert_eq!(true, result1.is_zero());
        assert_eq!(false, result2.is_zero());
    }

    #[test]
    fn sum() {
        let list = [
            VectorRow::from([1, 10, 100]),
            VectorRow::from([2, 20, 200]),
            VectorRow::from([3, 30, 300]),
        ];
        let result: VectorRow<i32, _> = list.into_iter().sum();
        assert_eq!([6, 60, 600], result.values);
    }

    #[test]
    fn sum_ref() {
        let list = [
            VectorRow::from([1, 10, 100]),
            VectorRow::from([2, 20, 200]),
            VectorRow::from([3, 30, 300]),
        ];
        let result: VectorRow<i32, _> = list.iter().sum();
        assert_eq!([6, 60, 600], result.values);
    }

    #[test]
    fn add() {
        let vector1 = VectorRow::<i32, _>::from([2, 20]);
        let vector2 = VectorRow::<i32, _>::from([1, 10]);
        let result = vector1 + vector2;
        assert_eq!([3, 30], result.values);
    }

    #[test]
    fn add_assign() {
        let mut vector1 = VectorRow::<i32, _>::from([2, 20]);
        let vector2 = VectorRow::<i32, _>::from([1, 10]);
        vector1 += vector2;
        assert_eq!([3, 30], vector1.values);
    }

    #[test]
    fn sub() {
        let vector1 = VectorRow::<i32, _>::from([3, 30]);
        let vector2 = VectorRow::<i32, _>::from([1, 10]);
        let result = vector1 - vector2;
        assert_eq!([2, 20], result.values);
    }

    #[test]
    fn sub_assign() {
        let mut vector1 = VectorRow::<i32, _>::from([3, 30]);
        let vector2 = VectorRow::<i32, _>::from([1, 10]);
        vector1 -= vector2;
        assert_eq!([2, 20], vector1.values);
    }

    #[test]
    fn neg() {
        let vector = VectorRow::<i32, _>::from([3, 30]);
        let result = -vector;
        assert_eq!([-3, -30], result.values);
    }

    #[test]
    fn dot_product() {
        let vector1 = VectorRow::<i32, _>::from([2, 20]);
        let vector2 = VectorRow::<i32, _>::from([3, 30]);
        let result = vector1 * vector2;
        assert_eq!(606, result);
    }

    #[test]
    fn scalar_mul() {
        let vector = VectorRow::<i32, _>::from([1, 10]);
        let result = vector * 5;
        assert_eq!([5, 50], result.values);
    }

    #[test]
    fn scalar_mul_assign() {
        let mut vector = VectorRow::from([1, 10]);
        vector *= 5;
        assert_eq!([5, 50], vector.values);
    }

    #[test]
    fn elementwise() {
        let vector = VectorRow::from([1, 2, 3]);
        let result = vector.elementwise(|x| x * 2);
        assert_eq!([2, 4, 6], result.values);
    }

    #[test]
    fn elementwise_mut() {
        let mut vector = VectorRow::from([1, 2, 3]);
        vector.elementwise_mut(|x| x * 2);
        assert_eq!([2, 4, 6], vector.values);
    }

    #[test]
    fn elementwise_operation() {
        let vector1 = VectorRow::from([3, 30, 300]);
        let vector2 = VectorRow::from([2, 20, 200]);
        let result = vector1.elementwise_operation(&vector2, |x, y| x * y);
        assert_eq!([6, 600, 60000], result.values);
    }

    #[test]
    fn elementwise_operation_mut() {
        let mut vector1 = VectorRow::from([3, 30, 300]);
        let vector2 = VectorRow::from([2, 20, 200]);
        vector1.elementwise_operation_mut(&vector2, |x, y| x * y);
        assert_eq!([6, 600, 60000], vector1.values);
    }

    #[test]
    fn transpose() {
        let vector = VectorRow::from([0, 1, 2]);
        let result = vector.transpose();
        assert_eq!([0, 1, 2], result.values);
    }

    #[test]
    fn hermitian_conjugate() {
        let vector = VectorRow::from([Complex::new(0, 0), Complex::new(1, 0), Complex::new(0, 1)]);
        let result = vector.hermitian_conjugate();
        assert_eq!(
            [Complex::new(0, 0), Complex::new(1, 0), Complex::new(0, -1)],
            result.values
        );
    }
}
