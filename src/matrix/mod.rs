#[derive(Debug, PartialEq)]
pub struct Matrix<T: Clone + Copy> {
    values: Vec<T>,
    size: (usize, usize), 
}

impl<T: Clone + Copy> Matrix<T> {
    pub fn from_value(value: T, size: (usize, usize)) -> Self {
        Self {values: vec![value; size.0 * size.1], size}
    }

    pub fn from_vec(values: Vec<T>, size: (usize, usize)) -> Self {
        if size.0 * size.1 != values.len() {
            panic!("values has wrong size, expected {} * {} = {}, received {}", size.0, size.1, size.0 * size.1, values.len());
        }

        Self {values, size}
    }

    pub fn from_array(values: &[T], size: (usize, usize)) -> Self {
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
        fn from_array() {
            let result = Matrix::from_array(&[0, 1, 2, 3], (2, 2));
            assert_eq!((2, 2), result.size);
            assert_eq!(4, result.values.len());
            assert_eq!(vec![0, 1, 2, 3], result.values);
        }

        #[test]
        #[should_panic(expected = "values has wrong size")]
        fn from_array_panic() {
            let _result = Matrix::from_array(&[0, 1, 2], (2, 2));
        }

        #[test]
        fn from_array_using_vec() {
            let result = Matrix::from_array(&vec![0, 1, 2, 3], (2, 2));
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
            let result = Matrix::from_array(&[0, 1, 2, 3, 4, 5], (2, 3));
            assert_eq!((2, 3), result.get_size());
        }

        #[test]
        fn get_vec() {
            let result = Matrix::from_array(&[0, 1, 2, 3, 4, 5], (2, 3));
            assert_eq!(vec![0, 1, 2, 3, 4, 5], result.get_vec());
        }

        #[test]
        fn to_vec() {
            let result = Matrix::from_array(&[0, 1, 2, 3, 4, 5], (2, 3));
            assert_eq!(vec![0, 1, 2, 3, 4, 5], result.to_vec());
        }

        #[test]
        fn unwrap() {
            let result = Matrix::from_array(&[0, 1, 2, 3, 4, 5], (2, 3));
            assert_eq!((vec![0, 1, 2, 3, 4, 5], (2, 3)), result.unwrap());
        }    }
}
