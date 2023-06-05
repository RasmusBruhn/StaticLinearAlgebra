use crate::operators;

#[derive(Debug, PartialEq)]
pub struct Matrix<T> 
where
    T: Clone,
    T: Copy,
    T: std::ops::Add<Output = T>,
    T: std::ops::Mul<Output = T>,
    T: std::iter::Sum,
{
    pub(crate) values: Vec<T>,
    pub(crate) size: (usize, usize), 
}

impl<T> Matrix<T>
where
    T: Clone,
    T: Copy,
    T: std::ops::Add<Output = T>,
    T: std::ops::Mul<Output = T>,
    T: std::iter::Sum,
{
    pub fn from_value(value: T, size: (usize, usize)) -> Self {
        Self {values: vec![value; size.0 * size.1], size}
    }

    pub fn from_vec(values: Vec<T>, size: (usize, usize)) -> Self {
        if size.0 * size.1 != values.len() {
            panic!("values has wrong size, expected {} * {} = {}, received {}", size.0, size.1, size.0 * size.1, values.len());
        }

        Self {values, size}
    }

    pub fn from_arr(values: &[T], size: (usize, usize)) -> Self {
        if size.0 * size.1 != values.len() {
            panic!("values has wrong size, expected {} * {} = {}, received {}", size.0, size.1, size.0 * size.1, values.len());
        }

        Self {values: values.to_vec(), size}
    }

    pub fn from_diag(values: &[T], zero_value: T) -> Self {
        let size = values.len();
        let mut use_values: Vec<T> = vec![zero_value; size * size];

        for (n, value) in values.iter().enumerate() {
            use_values[n * (size + 1)] = *value;
        }

        Self {values: use_values, size: (size, size)}
    }

    pub fn get_size(&self) -> (usize, usize) {
        self.size
    }

    pub fn get_vec(&self) -> Vec<T> {
        self.values.clone()
    }

    pub fn to_vec(self) -> Vec<T> {
        self.values
    }

    pub fn unwrap(self) -> (Vec<T>, (usize, usize)) {
        (self.values, self.size)
    }

    pub fn get_value(&self, row: usize, column: usize) -> T {
        if row >= self.size.0 {
            panic!("row {} is out of bound of width {}", row, self.size.0);
        }

        if column >= self.size.1 {
            panic!("column {} is out of bound of height {}", column, self.size.1);
        }

        self.values[column + self.size.1 * row]
    }

    pub fn set_value(&mut self, value: T, row: usize, column: usize) {
        if row >= self.size.0 {
            panic!("row {} is out of bound of width {}", row, self.size.0);
        }

        if column >= self.size.1 {
            panic!("column {} is out of bound of height {}", column, self.size.1);
        }

        self.values[column + self.size.1 * row] = value;
    }
}

impl<T> std::ops::Add for Matrix<T>
where
    T: Clone,
    T: Copy,
    T: std::ops::Add<Output = T>,
    T: std::ops::Mul<Output = T>,
    T: std::iter::Sum,
{
    type Output = Matrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        operators::add::matrix(&self, &rhs).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod matrix {
        use super::*;

        #[test]
        fn from_value() {
            let result: Matrix<f32> = Matrix::from_value(0., (2, 5));
            assert_eq!((2, 5), result.size);
            assert_eq!(10, result.values.len());

            for value in result.values {
                assert_eq!(0., value);
            }
        }

        #[test]
        fn from_vec() {
            let result = Matrix::from_vec(vec![0, 1, 2, 3], (2, 2));
            assert_eq!((2, 2), result.size);
            assert_eq!(4, result.values.len());
            assert_eq!(vec![0, 1, 2, 3], result.values);
        }

        #[test]
        #[should_panic(expected = "values has wrong size")]
        fn from_vec_panic() {
            let _result = Matrix::from_vec(vec![0, 1, 2], (2, 2));
        }

        #[test]
        fn from_arr() {
            let result = Matrix::from_arr(&[0, 1, 2, 3], (2, 2));
            assert_eq!((2, 2), result.size);
            assert_eq!(4, result.values.len());
            assert_eq!(vec![0, 1, 2, 3], result.values);
        }

        #[test]
        #[should_panic(expected = "values has wrong size")]
        fn from_arr_panic() {
            let _result = Matrix::from_arr(&[0, 1, 2], (2, 2));
        }

        #[test]
        fn from_arr_using_vec() {
            let result = Matrix::from_arr(&vec![0, 1, 2, 3], (2, 2));
            assert_eq!((2, 2), result.size);
            assert_eq!(4, result.values.len());
            assert_eq!(vec![0, 1, 2, 3], result.values);
        }

        #[test]
        fn from_diag_int() {
            let result = Matrix::from_diag(&[5, 2, 3], 0);
            assert_eq!((3, 3), result.size);
            assert_eq!(9, result.values.len());
            assert_eq!(vec![5, 0, 0, 0, 2, 0, 0, 0, 3], result.values);
        }

        #[test]
        fn from_diag_float() {
            let result = Matrix::from_diag(&vec![2., 1.5, 0.1], 0.);
            assert_eq!((3, 3), result.size);
            assert_eq!(9, result.values.len());
            assert_eq!(vec![2., 0., 0., 0., 1.5, 0., 0., 0., 0.1], result.values);
        }

        #[test]
        fn get_size() {
            let result = Matrix::from_arr(&[0, 1, 2, 3, 4, 5], (2, 3));
            assert_eq!((2, 3), result.get_size());
        }

        #[test]
        fn get_vec() {
            let result = Matrix::from_arr(&[0, 1, 2, 3, 4, 5], (2, 3));
            assert_eq!(vec![0, 1, 2, 3, 4, 5], result.get_vec());
        }

        #[test]
        fn to_vec() {
            let result = Matrix::from_arr(&[0, 1, 2, 3, 4, 5], (2, 3));
            assert_eq!(vec![0, 1, 2, 3, 4, 5], result.to_vec());
        }

        #[test]
        fn unwrap() {
            let result = Matrix::from_arr(&[0, 1, 2, 3, 4, 5], (2, 3));
            assert_eq!((vec![0, 1, 2, 3, 4, 5], (2, 3)), result.unwrap());
        }

        #[test]
        fn get_value() {
            let result = Matrix::from_arr(&[0, 1, 2, 3], (2, 2));
            assert_eq!(0, result.get_value(0, 0));
            assert_eq!(1, result.get_value(0, 1));
            assert_eq!(2, result.get_value(1, 0));
            assert_eq!(3, result.get_value(1, 1));
        }

        #[test]
        #[should_panic(expected = "is out of bound of width")]
        fn get_value_panic_row() {
            let result = Matrix::from_arr(&[0, 1, 2, 3], (2, 2));
            result.get_value(2, 0);
        }

        #[test]
        #[should_panic(expected = "is out of bound of height")]
        fn get_value_panic_column() {
            let result = Matrix::from_arr(&[0, 1, 2, 3], (2, 2));
            result.get_value(0, 2);
        }

        #[test]
        fn set_value() {
            let mut result = Matrix::from_arr(&[0, 1, 2, 3], (2, 2));
            result.set_value(10, 0, 0);
            result.set_value(11, 0, 1);
            result.set_value(12, 1, 0);
            result.set_value(13, 1, 1);
            assert_eq!(vec![10, 11, 12, 13], result.to_vec());
        }

        #[test]
        #[should_panic(expected = "is out of bound of width")]
        fn set_value_panic_row() {
            let mut result = Matrix::from_arr(&[0, 1, 2, 3], (2, 2));
            result.set_value(0, 2, 0);
        }

        #[test]
        #[should_panic(expected = "is out of bound of height")]
        fn set_value_panic_column() {
            let mut result = Matrix::from_arr(&[0, 1, 2, 3], (2, 2));
            result.set_value(0, 0, 2);
        }

        #[test]
        fn add_matrix() {
            let matrix1 = Matrix::from_arr(&[0., 1., 2., 3., 4., 5.], (3, 2));
            let matrix2 = Matrix::from_arr(&[0., 10., 20., 30., 40., 50.], (3, 2));
            let result = matrix1 + matrix2;
            assert_eq!((3, 2), result.get_size());
            assert_eq!(vec![0., 11., 22., 33., 44., 55.], result.to_vec());
        }   

        #[test]
        #[should_panic(expected = "Matrices must have identical size when adding")]
        fn add_matrix_panic() {
            let matrix1 = Matrix::from_arr(&[0., 1., 2., 3., 4., 5.], (3, 2));
            let matrix2 = Matrix::from_arr(&[0., 10., 20., 30., 40., 50.], (2, 3));
            let _result = matrix1 + matrix2;
        }   
    }
}
