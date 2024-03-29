//! This part of the library includes all definitions of matrices and vectors
//! and all common operations of them

mod matrix;
mod vector_column;
mod vector_row;

pub use matrix::Matrix;
pub use vector_column::VectorColumn;
pub use vector_row::VectorRow;