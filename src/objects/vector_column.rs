use std::ops::{Index, IndexMut, Add, Sub, AddAssign, SubAssign, Mul};
use num::traits::{Zero, Num};
use std::iter::Sum;

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

    fn mul(self, rhs: VectorColumn<TR, S>) -> Self::Output {
        (0..S).map(|i| self[i] * rhs[i]).sum()
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

    fn mul(self, rhs: TR) -> Self::Output {
        let values: [TO; S] = match (0..S).map(|i| self[i] * rhs).collect::<Vec<TO>>().try_into() {
            Ok(result) => result,
            Err(_) => panic!("Should not happen"),
        };

        Self::Output {values}
    }
}

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
    fn scalar_mul() {
        let vector = VectorColumn::new(&[0, 1, 2]);
        let result = vector * 5;
        assert_eq!([0, 5, 10], result.values);
    }
}
