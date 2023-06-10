use std::ops::{Index, IndexMut, Add, Sub, AddAssign, SubAssign, Mul, MulAssign};
use num::{traits::{Zero, Num}, Complex};
use std::iter::Sum;
use super::VectorColumn;

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
    pub fn new(values: &[T; S]) -> Self {
        Self {values: *values}
    }

    pub fn from_value(value: T) -> Self {
        Self {values: [value; S]}
    }

    pub fn get_values(&self) -> &[T; S] {
        &self.values
    }

    pub fn get_values_mut(&mut self) -> &mut [T; S] {
        &mut self.values
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
        (0..S).any(|i| self[i] != T::zero()) ^ true
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
            result = result + value;
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
            result = result + *value;
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

    fn add(self, rhs: VectorRow<TR, S>) -> Self::Output {
        let values: [TO; S] = match (0..S).map(|i| self[i] + rhs[i]).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<TL, TR, const S: usize> AddAssign<VectorRow<TR, S>> for VectorRow<TL, S>
where
    TL: Copy,
    TL: Add<TR, Output = TL>,
    TR: Copy,
{
    fn add_assign(&mut self, rhs: VectorRow<TR, S>) {
        let values: [TL; S] = match (0..S).map(|i| self[i] + rhs[i]).collect::<Vec<TL>>().try_into() {
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

    fn sub(self, rhs: VectorRow<TR, S>) -> Self::Output {
        let values: [TO; S] = match (0..S).map(|i| self[i] - rhs[i]).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<TL, TR, const S: usize> SubAssign<VectorRow<TR, S>> for VectorRow<TL, S>
where
    TL: Copy,
    TL: Sub<TR, Output = TL>,
    TR: Copy,
{
    fn sub_assign(&mut self, rhs: VectorRow<TR, S>) {
        let values: [TL; S] = match (0..S).map(|i| self[i] - rhs[i]).collect::<Vec<TL>>().try_into() {
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

    fn mul(self, rhs: VectorRow<TR, S>) -> Self::Output {
        (0..S).map(|i| self[i] * rhs[i]).sum()
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

    fn mul(self, rhs: VectorColumn<TR, S>) -> Self::Output {
        (0..S).map(|i| self[i] * rhs[i]).sum()
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

    fn mul(self, rhs: TR) -> Self::Output {
        let values: [TO; S] = match (0..S).map(|i| self[i] * rhs).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

impl<TL, TR, const S: usize> MulAssign<TR> for VectorRow<TL, S>
where
    TL: Copy,
    TL: Mul<TR, Output = TL>,
    TR: Copy,
    TR: Num,
{
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
        impl<TR, TO, const S: usize> Mul<VectorRow<TR, S>> for $TL
        where
            $TL: Mul<TR, Output = TO>,
            TR: Copy,
            TO: Copy,
        {
            type Output = VectorRow<TO, S>;

            fn mul(self, rhs: VectorRow<TR, S>) -> Self::Output {
                let values: [TO; S] = match (0..S).map(|i| self * rhs[i]).collect::<Vec<TO>>().try_into() {
                    Ok(result) => result,
                    Err(_) => panic!("Should not happen"),
                };

                Self::Output {values}
            }
        }
    };
}

impl<T, TR, TO, const S: usize> Mul<VectorRow<TR, S>> for Complex<T>
where
    Complex<T>: Copy,
    Complex<T>: Mul<TR, Output = TO>,
    TR: Copy,
    TO: Copy,
{
    type Output = VectorRow<TO, S>;

    fn mul(self, rhs: VectorRow<TR, S>) -> Self::Output {
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
        let result = VectorRow::new(&[0, 1, 2, 3]);
        assert_eq!([0, 1, 2, 3], result.values);
    }

    #[test]
    fn from_value() {
        let result: VectorRow<f64, 5> = VectorRow::from_value(5.6);
        assert_eq!([5.6, 5.6, 5.6, 5.6, 5.6], result.values);
    }

    #[test]
    fn get_values() {
        let result = VectorRow::new(&[0, 1, 2]);
        assert_eq!([0, 1, 2], *result.get_values());
    }

    #[test]
    fn get_values_mut() {
        let mut result = VectorRow::new(&[0, 1, 2]);
        let data = result.get_values_mut();
        data[1] = 5;
        assert_eq!([0, 5, 2], result.values);
    }

    #[test]
    fn index() {
        let result = VectorRow::new(&[0, 1, 2]);
        assert_eq!(1, result[1]);
    }

    #[test]
    fn index_mut() {
        let mut result = VectorRow::new(&[0, 1, 2]);
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
        let result1 = VectorRow::new(&[0, 0, 0]);
        let result2 = VectorRow::new(&[0, 1, 0]);
        assert_eq!(true, result1.is_zero());
        assert_eq!(false, result2.is_zero());
    }

    #[test]
    fn sum() {
        let list: [VectorRow<i32, 3>; 3] = [VectorRow::new(&[0, 1, 2]), VectorRow::new(&[0, 10, 20]), VectorRow::new(&[0, 100, 200])];
        let result: VectorRow<i32, 3> = list.into_iter().sum();
        assert_eq!([0, 111, 222], result.values);
    }

    #[test]
    fn sum_ref() {
        let list: [VectorRow<i32, 3>; 3] = [VectorRow::new(&[0, 1, 2]), VectorRow::new(&[0, 10, 20]), VectorRow::new(&[0, 100, 200])];
        let result: VectorRow<i32, 3> = list.iter().sum();
        assert_eq!([0, 111, 222], result.values);
    }

    #[test]
    fn add() {
        let vector1 = VectorRow::new(&[0, 1, 2]);
        let vector2 = VectorRow::new(&[0, 10, 20]);
        let result = vector1 + vector2;
        assert_eq!([0, 11, 22], result.values);
    }

    #[test]
    fn add_assign() {
        let mut vector1 = VectorRow::new(&[0, 1, 2]);
        let vector2 = VectorRow::new(&[0, 10, 20]);
        vector1 += vector2;
        assert_eq!([0, 11, 22], vector1.values);
    }

    #[test]
    fn sub() {
        let vector1 = VectorRow::new(&[0, 1, 2]);
        let vector2 = VectorRow::new(&[0, 10, 20]);
        let result = vector1 - vector2;
        assert_eq!([0, -9, -18], result.values);
    }

    #[test]
    fn sub_assign() {
        let mut vector1 = VectorRow::new(&[0, 1, 2]);
        let vector2 = VectorRow::new(&[0, 10, 20]);
        vector1 -= vector2;
        assert_eq!([0, -9, -18], vector1.values);
    }

    #[test]
    fn dot_product() {
        let vector1 = VectorRow::new(&[0, 1, 2]);
        let vector2 = VectorRow::new(&[3, 4, 5]);
        let result = vector1 * vector2;
        assert_eq!(14, result);
    }

    #[test]
    fn dot_product_column() {
        let vector1 = VectorRow::new(&[0, 1, 2]);
        let vector2 = VectorColumn::new(&[3, 4, 5]);
        let result = vector1 * vector2;
        assert_eq!(14, result);
    }

    #[test]
    fn scalar_mul_right() {
        let vector = VectorRow::new(&[0, 1, 2]);
        let result = vector * 5;
        assert_eq!([0, 5, 10], result.values);
    }

    #[test]
    fn scalar_mul_assign() {
        let mut vector = VectorRow::new(&[0, 1, 2]);
        vector *= 5;
        assert_eq!([0, 5, 10], vector.values);
    }

    macro_rules! dot_method_test {
        ($T:ty, $name:ident) => {
            #[test]
            fn $name() {
                let vector: VectorRow<$T, 3> = VectorRow::new(&[0 as $T, 1 as $T, 2 as $T]);
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
        let vector = VectorRow::new(&[Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 1.)]);
        let result = Complex::new(0., 4.) * vector;
        assert_eq!([Complex::new(0., 0.), Complex::new(0., 4.), Complex::new(-4., 0.)], result.values);    
    }
}
