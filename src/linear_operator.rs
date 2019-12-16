#![feature(specialization)]
#![allow(non_camel_case_types)]
use ndarray::{prelude::*, ArrayBase, Data, DataMut};
type c64 = num::complex::Complex64;
type c32 = num::complex::Complex32;

/// A trait to generalize over 1-norm estimates of a matrix `A`, matrix powers `A^m`,
/// or matrix products `A1 * A2 * ... * An`.
///
/// In the 1-norm estimator, one repeatedly constructs a matrix-matrix product between some n×n
/// matrix X and some other n×t matrix Y. If one wanted to estimate the 1-norm of a matrix m times
/// itself, X^m, it might thus be computationally less expensive to repeatedly apply
/// X * ( * ( X ... ( X * Y ) rather than to calculate Z = X^m = X * X * ... * X and then apply Z *
/// Y. In the first case, one has several matrix-matrix multiplications with complexity O(m*n*n*t),
/// while in the latter case one has O(m*n*n*n) (plus one more O(n*n*t)).
///
/// So in case of t << n, it is cheaper to repeatedly apply matrix multiplication to the smaller
/// matrix on the RHS, rather than to construct one definite matrix on the LHS.  Of course, this is
/// modified by the number of iterations needed when performing the norm estimate, sustained
/// performance of the matrix multiplication method used, etc.
///
/// It is at the designation of the user to check what is more efficient: to pass in one definite
/// matrix or choose the alternative route described here.
pub(crate) trait LinearOperator<T1: num::Num> {
    fn multiply_matrix<S, T2: num::Num>(
        &self,
        b: &mut ArrayBase<S, Ix2>,
        c: &mut ArrayBase<S, Ix2>,
        transpose: bool,
    ) where
        S: DataMut<Elem = T2>;
}

trait MatrixMultiplication<T: num::Num> {
    unsafe fn muliply(
        &self,
        layout: cblas::Layout,
        mat_b: &[T],
        result: &mut [T],
        n: i32,
        t: i32,
        a_transpose: cblas::Transpose,
        b_transpose: cblas::Transpose,
    ) {
        panic!("Number type not implement")
    }
}

unsafe fn muliply<S1: MatrixMultiplication<T2>, T2: num::Num>(
    _: S1,
    layout: cblas::Layout,
    mat_b: &[T2],
    result: &mut [T2],
    n: i32,
    t: i32,
    a_transpose: cblas::Transpose,
    b_transpose: cblas::Transpose,
) {
    panic!("Number type not implement")
}

impl MatrixMultiplication<f64> for &[f64] {
    unsafe fn muliply(
        &self,
        layout: cblas::Layout,
        mat_b: &[f64],
        result: &mut [f64],
        n: i32,
        t: i32,
        a_transpose: cblas::Transpose,
        b_transpose: cblas::Transpose,
    ) {
        unsafe {
            cblas::dgemm(
                layout,
                a_transpose,
                b_transpose,
                n,
                t,
                n,
                1.0,
                &self,
                n,
                mat_b,
                t,
                0.0,
                result,
                t,
            )
        }
    }
}

impl MatrixMultiplication<f32> for &[f32] {
    unsafe fn muliply(
        &self,
        layout: cblas::Layout,
        mat_b: &[f32],
        result: &mut [f32],
        n: i32,
        t: i32,
        a_transpose: cblas::Transpose,
        b_transpose: cblas::Transpose,
    ) {
        unsafe {
            cblas::sgemm(
                layout,
                a_transpose,
                b_transpose,
                n,
                t,
                n,
                1.0,
                &self,
                n,
                mat_b,
                t,
                0.0,
                result,
                t,
            )
        }
    }
}

impl MatrixMultiplication<c64> for &[c64] {
    unsafe fn muliply(
        &self,
        layout: cblas::Layout,
        mat_b: &[c64],
        result: &mut [c64],
        n: i32,
        t: i32,
        a_transpose: cblas::Transpose,
        b_transpose: cblas::Transpose,
    ) {
        unsafe {
            cblas::zgemm(
                layout,
                a_transpose,
                b_transpose,
                n,
                t,
                n,
                c64::new(1.0, 0.0),
                &self,
                n,
                mat_b,
                t,
                c64::new(0.0, 0.0),
                result,
                t,
            )
        }
    }
}

impl MatrixMultiplication<c32> for &[c32] {
    unsafe fn muliply(
        &self,
        layout: cblas::Layout,
        mat_b: &[c32],
        result: &mut [c32],
        n: i32,
        t: i32,
        a_transpose: cblas::Transpose,
        b_transpose: cblas::Transpose,
    ) {
        unsafe {
            cblas::cgemm(
                layout,
                a_transpose,
                b_transpose,
                n,
                t,
                n,
                c32::new(1.0, 0.0),
                &self,
                n,
                mat_b,
                t,
                c32::new(0.0, 0.0),
                result,
                t,
            )
        }
    }
}

impl<S1, T1: num::Num> LinearOperator<T1> for ArrayBase<S1, Ix2>
where
    S1: Data<Elem = T1>,
{
    fn multiply_matrix<S2, T2: num::Num>(
        &self,
        b: &mut ArrayBase<S2, Ix2>,
        c: &mut ArrayBase<S2, Ix2>,
        transpose: bool,
    ) where
        S2: DataMut<Elem = T2>,
    {
        let (n_rows, n_cols) = self.dim();
        assert_eq!(
            n_rows, n_cols,
            "Number of rows and columns does not match: `self` has to be a square matrix"
        );
        let n = n_rows;

        let (b_n, b_t) = b.dim();
        let (c_n, c_t) = b.dim();

        assert_eq!(
            n, b_n,
            "Number of rows of b not equal to number of rows of `self`."
        );
        assert_eq!(
            n, c_n,
            "Number of rows of c not equal to number of rows of `self`."
        );

        assert_eq!(
            b_t, c_t,
            "Number of columns of b not equal to number of columns of c."
        );

        let t = b_t;

        let (a_slice, a_layout) =
            super::as_slice_with_layout(self).expect("Matrix `self` not contiguous.");
        let (b_slice, b_layout) =
            super::as_slice_with_layout(b).expect("Matrix `b` not contiguous.");
        let (c_slice, c_layout) =
            super::as_slice_with_layout_mut(c).expect("Matrix `c` not contiguous.");

        assert_eq!(a_layout, b_layout);
        assert_eq!(a_layout, c_layout);

        let layout = a_layout;

        let a_transpose = if transpose {
            cblas::Transpose::Ordinary
        } else {
            cblas::Transpose::None
        };

        unsafe {
            a_slice.muliply(
                layout,
                b_slice,
                c_slice,
                n as i32,
                t as i32,
                a_transpose,
                cblas::Transpose::None,
            );
        }
    }
}

impl<S1, T1> LinearOperator<T1> for [&ArrayBase<S1, Ix2>]
where
    S1: Data<Elem = T1>,
    T1: num::Num,
{
    fn multiply_matrix<S2, T2>(
        &self,
        b: &mut ArrayBase<S2, Ix2>,
        c: &mut ArrayBase<S2, Ix2>,
        transpose: bool,
    ) where
        S2: DataMut<Elem = T2>,
        T2: num::Num,
    {
        if self.len() > 0 {
            let mut reversed;
            let mut forward;

            // TODO: Investigate, if an enum instead of a trait object might be more performant.
            // This probably doesn't matter for large matrices, but could have a measurable impact
            // on small ones.
            let a_iter: &mut dyn DoubleEndedIterator<Item = _> = if transpose {
                reversed = self.iter().rev();
                &mut reversed
            } else {
                forward = self.iter();
                &mut forward
            };
            let a = a_iter.next().unwrap(); // Ok because of if condition
            a.multiply_matrix::<S2, T2>(b, c, transpose);

            // NOTE: The swap in the loop body makes use of the fact that in all instances where
            // `multiply_matrix` is used, the values potentially stored in `b` are not required
            // anymore.
            for a in a_iter {
                std::mem::swap(b, c);
                a.multiply_matrix(b, c, transpose);
            }
        }
    }
}

impl<S1, T1> LinearOperator<T1> for (&ArrayBase<S1, Ix2>, usize)
where
    S1: Data<Elem = T1>,
    T1: num::Num,
{
    fn multiply_matrix<S2, T2>(
        &self,
        b: &mut ArrayBase<S2, Ix2>,
        c: &mut ArrayBase<S2, Ix2>,
        transpose: bool,
    ) where
        S2: DataMut<Elem = T2>,
        T2: num::Num,
    {
        let a = self.0;
        let m = self.1;
        if m > 0 {
            a.multiply_matrix(b, c, transpose);
            for _ in 1..m {
                std::mem::swap(b, c);
                self.0.multiply_matrix(b, c, transpose);
            }
        }
    }
}
