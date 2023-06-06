use std::ops::{Index, IndexMut, Add};

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
{
    pub fn from_diag(values: &[T; S], zero_value: T) -> Self {
        let mut use_values= [[zero_value; S]; S];
        
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

impl<T, TR, O, const R: usize, const C: usize> Add<Matrix<TR, R, C>> for Matrix<T, R, C>
where
    T: Copy,
    T: std::ops::Add<TR, Output = O>,
    TR: Copy,
    O: Copy,
{
    type Output = Matrix<O, R, C>;

    fn add(self, rhs: Matrix<TR, R, C>) -> Self::Output {
        let values: [[O; C]; R] = match self.values.iter().zip(rhs.values.iter())
        .map(|(lhs, rhs)| match lhs.iter().zip(rhs.iter())
        .map(|(lhs, rhs)| *lhs + *rhs).collect::<Vec<O>>().try_into() {
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
    for<'a> &'a T: std::ops::Add<&'a TR, Output = O>,
    TR: Copy,
    O: Copy,
{
    type Output = Matrix<O, R, C>;

    fn add(self, rhs: &Matrix<TR, R, C>) -> Self::Output {
        let values: [[O; C]; R] = match self.values.iter().zip(rhs.values.iter())
        .map(|(lhs, rhs)| match lhs.iter().zip(rhs.iter())
        .map(|(lhs, rhs)| lhs + rhs).collect::<Vec<O>>().try_into() {
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
            let result = Matrix::from_diag(&[1, 2], 0);
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
    }
}
