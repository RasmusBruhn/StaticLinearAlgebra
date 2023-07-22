//! This library is a static sized linear algebra library.
//! 
//! It contains matrices and vectors whose sizes must all be known at compile time
//! and allows a multiplication and other operation between these.
//! All size compatibilities are checkedd at compile time.

mod objects;

pub use objects::Matrix;
pub use objects::VectorColumn;
pub use objects::VectorColumn as Vector;
pub use objects::VectorRow;