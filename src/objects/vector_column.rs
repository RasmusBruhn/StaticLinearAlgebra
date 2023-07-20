use std::ops::{Index, IndexMut, Add, Sub, AddAssign, SubAssign, Mul, MulAssign};
use num::{traits::{Zero, Num}, Complex};
use std::iter::Sum;
use core::ops::Neg;
use super::{Matrix, VectorRow};

/// A static column vector type
/// 
/// Size must be known at compile time but operations are checked for size compatibility at compile time too
/// 
/// S: The length of the vector
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct VectorColumn<T, const S: usize>
where
    T: Copy,
{
    pub(crate) values: [T; S],
}

impl<T, const S: usize> VectorColumn<T, S>
where
    T: Copy,
{
    /// Initializes a new column vector with the given values
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = linear_algebra::VectorColumn::new(&[0, 1, 2]);
    /// 
    /// assert_eq!(&[0, 1, 2], x.get_values());
    /// ```
    pub fn new(values: &[T; S]) -> Self {
        Self {values: *values}
    }

    /// Initializes a new column vector filled with a single value
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = linear_algebra::VectorColumn::<f32, 2>::from_value(1.);
    /// 
    /// assert_eq!(&[1., 1.], x.get_values());
    /// ```
    pub fn from_value(value: T) -> Self {
        Self {values: [value; S]}
    }

    /// Retrieves a reference to the data of the column vector
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = linear_algebra::VectorColumn::new(&[0, 1]);
    /// let data = x.get_values();
    /// 
    /// assert_eq!(&[0, 1], data);
    /// ```
    pub fn get_values(&self) -> &[T; S] {
        &self.values
    }

    /// Retrieves a mutable reference to the data of the column vector
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = linear_algebra::VectorColumn::new(&[0, 1]);
    /// let data = x.get_values_mut();
    /// data[0] = 5;
    /// 
    /// assert_eq!(&[5, 1], x.get_values());
    /// ```
    pub fn get_values_mut(&mut self) -> &mut [T; S] {
        &mut self.values
    }

    /// Transposes the column vector into a row vector
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = linear_algebra::VectorColumn::new(&[0, 1, 2]);
    /// let y = x.transpose();
    /// 
    /// assert_eq!(&[0, 1, 2], y.get_values());
    /// ```
    pub fn transpose(&self) -> VectorRow<T, S> {
        VectorRow {values: self.values}
    }
}

impl<T, const S: usize> VectorColumn<Complex<T>, S> 
where
    T: Copy,
    T: Num,
    T: Neg<Output = T>,
{
    /// Takes the hermitian conjugate of the column vector (transpose the vector 
    /// and complex conjugate each element (change the sign of the imaginary part))
    /// 
    /// # Examples
    /// 
    /// ```
    /// use num::Complex;
    /// 
    /// let x = linear_algebra::VectorColumn::new(&[Complex::new(1, 0), Complex::new(0, 2)]);
    /// let y = x.hermitian_conjugate();
    /// 
    /// assert_eq!(&[Complex::new(1, 0), Complex::new(0, -2)], y.get_values())
    /// ```
    pub fn hermitian_conjugate(&self) -> VectorRow<Complex<T>, S> {
        let values: [Complex<T>; S] = match (0..S).map(|i| self[i].conj()).collect::<Vec<Complex<T>>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        VectorRow {values}
    }
}

impl<T, const S: usize> Index<usize> for VectorColumn<T, S> 
where
    T: Copy,
{
    type Output = T;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.values[idx]
    }
}

impl<T, const S: usize> IndexMut<usize> for VectorColumn<T, S> 
where
    T: Copy,
{
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.values[idx]
    }
}

impl<T, const S: usize> Zero for VectorColumn<T, S>
where
    T: Copy,
    T: Zero,
    T: PartialEq,
{
    fn zero() -> Self {
        Self::from_value(T::zero())
    }

    fn is_zero(&self) -> bool {
        (0..S).any(|i| self[i] != T::zero()) ^ true
    }
}

impl<T, const S: usize> Sum for VectorColumn<T, S>
where
    T: Copy,
    T: Zero,
    T: Add<T, Output = T>,
    T: PartialEq,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut result: Self = VectorColumn::zero();

        for value in iter {
            result = result + value;
        }

        result
    }
}

impl<'a, T, const S: usize> Sum<&'a VectorColumn<T, S>> for VectorColumn<T, S>
where
    T: Copy,
    T: Zero,
    T: Add<T, Output = T>,
    T: PartialEq,
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let mut result: Self = VectorColumn::zero();

        for value in iter {
            result = result + *value;
        }

        result
    }
}

impl<TL, TR, TO, const S: usize> Add<VectorColumn<TR, S>> for VectorColumn<TL, S>
where
    TL: Copy,
    TL: Add<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
{
    type Output = VectorColumn<TO, S>;

    /// Normal elementwise addition of two column vectors
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = linear_algebra::VectorColumn::new(&[0, 1]);
    /// let y = linear_algebra::VectorColumn::new(&[0, 10]);
    /// 
    /// let z = x + y;
    /// 
    /// assert_eq!(&[0, 11], z.get_values());
    /// ```
    fn add(self, rhs: VectorColumn<TR, S>) -> Self::Output {
        let values: [TO; S] = match (0..S).map(|i| self[i] + rhs[i]).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<TL, TR, const S: usize> AddAssign<VectorColumn<TR, S>> for VectorColumn<TL, S>
where
    TL: Copy,
    TL: Add<TR, Output = TL>,
    TR: Copy,
{
    /// Normal elementwise addition of two column vectors
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut x = linear_algebra::VectorColumn::new(&[0, 1]);
    /// let y = linear_algebra::VectorColumn::new(&[0, 10]);
    /// 
    /// x += y;
    /// 
    /// assert_eq!(&[0, 11], x.get_values());
    /// ```
    fn add_assign(&mut self, rhs: VectorColumn<TR, S>) {
        let values: [TL; S] = match (0..S).map(|i| self[i] + rhs[i]).collect::<Vec<TL>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        self.values = values;
    }
}

impl<TL, TR, TO, const S: usize> Sub<VectorColumn<TR, S>> for VectorColumn<TL, S>
where
    TL: Copy,
    TL: Sub<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
{
    type Output = VectorColumn<TO, S>;

    /// Normal elementwise subtraction of two matrices
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = linear_algebra::VectorColumn::new(&[0, 1]);
    /// let y = linear_algebra::VectorColumn::new(&[0, 10]);
    /// 
    /// let z = x - y;
    /// 
    /// assert_eq!(&[0, -9], z.get_values());
    /// ```
    fn sub(self, rhs: VectorColumn<TR, S>) -> Self::Output {
        let values: [TO; S] = match (0..S).map(|i| self[i] - rhs[i]).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<TL, TR, const S: usize> SubAssign<VectorColumn<TR, S>> for VectorColumn<TL, S>
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
    /// let mut x = linear_algebra::VectorColumn::new(&[0, 1]);
    /// let y = linear_algebra::VectorColumn::new(&[0, 10]);
    /// 
    /// x -= y;
    /// 
    /// assert_eq!(&[0, -9], x.get_values());
    /// ```
    fn sub_assign(&mut self, rhs: VectorColumn<TR, S>) {
        let values: [TL; S] = match (0..S).map(|i| self[i] - rhs[i]).collect::<Vec<TL>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        self.values = values;
    }
}

impl<TL, TR, TO, const S: usize> Mul<VectorColumn<TR, S>> for VectorColumn<TL, S>
where
    TL: Copy,
    TL: Mul<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
    TO: Sum,
{
    type Output = TO;

    /// Inner product (dot product) between two column vectors
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = linear_algebra::VectorColumn::new(&[0, 1]);
    /// let y = linear_algebra::VectorColumn::new(&[0, 10]);
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(10, z);
    /// ```
    fn mul(self, rhs: VectorColumn<TR, S>) -> Self::Output {
        (0..S).map(|i| self[i] * rhs[i]).sum()
    }
}

impl<TL, TR, TO, const R: usize, const C: usize> Mul<VectorRow<TR, C>> for VectorColumn<TL, R>
where
    TL: Copy,
    TL: Mul<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
    TO: Sum,
{
    type Output = Matrix<TO, R, C>;

    /// Outer product between a column vector and a row vector
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = linear_algebra::VectorColumn::new(&[0, 1]);
    /// let y = linear_algebra::VectorRow::new(&[10, 20]);
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(&[[0, 0], [10, 20]], z.get_values());
    /// ```
    fn mul(self, rhs: VectorRow<TR, C>) -> Self::Output {
        let values: [[TO; C]; R] = 
            match (0..R).map(|r| 
            match (0..C).map(|c| self[r] * rhs[c]).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        }).collect::<Vec<[TO; C]>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Matrix {values}
    }
}

impl<TL, TR, TO, const S: usize> Mul<TR> for VectorColumn<TL, S>
where
    TL: Copy,
    TL: Mul<TR, Output = TO>,
    TR: Copy,
    TR: Num,
    TO: Copy,
{
    type Output = VectorColumn<TO, S>;

    /// Scalar multiplication from the right, this is preferable from lhs scalar multiplication
    /// 
    /// # Examples
    /// 
    /// ```
    /// let x = linear_algebra::VectorColumn::new(&[0, 1]);
    /// let y = 10;
    /// 
    /// let z = x * y;
    /// 
    /// assert_eq!(&[0, 10], z.get_values());
    /// ```
    fn mul(self, rhs: TR) -> Self::Output {
        let values: [TO; S] = match (0..S).map(|i| self[i] * rhs).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<TL, TR, const S: usize> MulAssign<TR> for VectorColumn<TL, S>
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
    /// let mut x = linear_algebra::VectorColumn::new(&[0, 1]);
    /// let y = 10;
    /// 
    /// x *= y;
    /// 
    /// assert_eq!(&[0, 10], x.get_values());
    /// ```
    fn mul_assign(&mut self, rhs: TR) {
        let values: [TL; S] = match (0..S).map(|i| self[i] * rhs).collect::<Vec<TL>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        self.values = values;
    }
}

macro_rules! dot_method {
    ($TL:ty) => {
        impl<TR, TO, const S: usize> Mul<VectorColumn<TR, S>> for $TL
        where
            $TL: Mul<TR, Output = TO>,
            TR: Copy,
            TO: Copy,
        {
            type Output = VectorColumn<TO, S>;

            /// Scalar multiplication from the left, this only works for specific types, for generic types use rhs multiplication
            fn mul(self, rhs: VectorColumn<TR, S>) -> Self::Output {
                let values: [TO; S] = match (0..S).map(|i| self * rhs[i]).collect::<Vec<TO>>().try_into() {
                    Ok(result) => result,
                    Err(_) => panic!("Should not happen"),
                };

                Self::Output {values}
            }
        }
    };
}

impl<T, TR, TO, const S: usize> Mul<VectorColumn<TR, S>> for Complex<T>
where
    Complex<T>: Copy,
    Complex<T>: Mul<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
{
    type Output = VectorColumn<TO, S>;

    /// Scalar multiplication from the left, this only works for specific types, for generic types use rhs multiplication
    fn mul(self, rhs: VectorColumn<TR, S>) -> Self::Output {
        let values: [TO; S] = match (0..S).map(|i| self * rhs[i]).collect::<Vec<TO>>().try_into() {
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
        let result = VectorColumn::new(&[0, 1, 2, 3]);
        assert_eq!([0, 1, 2, 3], result.values);
    }

    #[test]
    fn from_value() {
        let result: VectorColumn<f64, 5> = VectorColumn::from_value(5.6);
        assert_eq!([5.6, 5.6, 5.6, 5.6, 5.6], result.values);
    }

    #[test]
    fn get_values() {
        let result = VectorColumn::new(&[0, 1, 2]);
        assert_eq!([0, 1, 2], *result.get_values());
    }

    #[test]
    fn get_values_mut() {
        let mut result = VectorColumn::new(&[0, 1, 2]);
        let data = result.get_values_mut();
        data[1] = 5;
        assert_eq!([0, 5, 2], result.values);
    }

    #[test]
    fn index() {
        let result = VectorColumn::new(&[0, 1, 2]);
        assert_eq!(1, result[1]);
    }

    #[test]
    fn index_mut() {
        let mut result = VectorColumn::new(&[0, 1, 2]);
        result[1] = 5;
        assert_eq!([0, 5, 2], result.values);
    }

    #[test]
    fn zero() {
        let result: VectorColumn<i32, 5> = VectorColumn::zero();
        assert_eq!([0, 0, 0, 0, 0], result.values);
    }

    #[test]
    fn is_zero() {
        let result1 = VectorColumn::new(&[0, 0, 0]);
        let result2 = VectorColumn::new(&[0, 1, 0]);
        assert_eq!(true, result1.is_zero());
        assert_eq!(false, result2.is_zero());
    }

    #[test]
    fn sum() {
        let list: [VectorColumn<i32, 3>; 3] = [VectorColumn::new(&[0, 1, 2]), VectorColumn::new(&[0, 10, 20]), VectorColumn::new(&[0, 100, 200])];
        let result: VectorColumn<i32, 3> = list.into_iter().sum();
        assert_eq!([0, 111, 222], result.values);
    }

    #[test]
    fn sum_ref() {
        let list: [VectorColumn<i32, 3>; 3] = [VectorColumn::new(&[0, 1, 2]), VectorColumn::new(&[0, 10, 20]), VectorColumn::new(&[0, 100, 200])];
        let result: VectorColumn<i32, 3> = list.iter().sum();
        assert_eq!([0, 111, 222], result.values);
    }

    #[test]
    fn add() {
        let vector1 = VectorColumn::new(&[0, 1, 2]);
        let vector2 = VectorColumn::new(&[0, 10, 20]);
        let result = vector1 + vector2;
        assert_eq!([0, 11, 22], result.values);
    }

    #[test]
    fn add_assign() {
        let mut vector1 = VectorColumn::new(&[0, 1, 2]);
        let vector2 = VectorColumn::new(&[0, 10, 20]);
        vector1 += vector2;
        assert_eq!([0, 11, 22], vector1.values);
    }

    #[test]
    fn sub() {
        let vector1 = VectorColumn::new(&[0, 1, 2]);
        let vector2 = VectorColumn::new(&[0, 10, 20]);
        let result = vector1 - vector2;
        assert_eq!([0, -9, -18], result.values);
    }

    #[test]
    fn sub_assign() {
        let mut vector1 = VectorColumn::new(&[0, 1, 2]);
        let vector2 = VectorColumn::new(&[0, 10, 20]);
        vector1 -= vector2;
        assert_eq!([0, -9, -18], vector1.values);
    }

    #[test]
    fn dot_product() {
        let vector1 = VectorColumn::new(&[0, 1, 2]);
        let vector2 = VectorColumn::new(&[3, 4, 5]);
        let result = vector1 * vector2;
        assert_eq!(14, result);
    }

    #[test]
    fn out_product() {
        let vector_row = VectorRow::new(&[0, 1]);
        let vector_column = VectorColumn::new(&[2, 3, 4]);
        let result = vector_column * vector_row;
        assert_eq!([[0, 2], [0, 3], [0, 4]], result.values);
    }

    #[test]
    fn scalar_mul_right() {
        let vector = VectorColumn::new(&[0, 1, 2]);
        let result = vector * 5;
        assert_eq!([0, 5, 10], result.values);
    }

    #[test]
    fn scalar_mul_assign() {
        let mut vector = VectorColumn::new(&[0, 1, 2]);
        vector *= 5;
        assert_eq!([0, 5, 10], vector.values);
    }

    macro_rules! dot_method_test {
        ($T:ty, $name:ident) => {
            #[test]
            fn $name() {
                let vector: VectorColumn<$T, 3> = VectorColumn::new(&[0 as $T, 1 as $T, 2 as $T]);
                let result = (4 as $T) * vector;
                assert_eq!([0 as $T, 4 as $T, 8 as $T], result.values);    
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
        let vector = VectorColumn::new(&[Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 1.)]);
        let result = Complex::new(0., 4.) * vector;
        assert_eq!([Complex::new(0., 0.), Complex::new(0., 4.), Complex::new(-4., 0.)], result.values);    
    }

    #[test]
    fn transpose() {
        let vector = VectorColumn::new(&[0, 1, 2]);
        let result = vector.transpose();
        assert_eq!([0, 1, 2], result.values);
    }

    #[test]
    fn hermitian_conjugate() {
        let vector = VectorColumn::new(&[Complex::new(0, 0), Complex::new(1, 0), Complex::new(0, 1)]);
        let result = vector.hermitian_conjugate();
        assert_eq!([Complex::new(0, 0), Complex::new(1, 0), Complex::new(0, -1)], result.values);
    }
}
