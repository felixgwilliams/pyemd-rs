use numpy::ndarray::prelude::*;

use crate::common::get_cow_slice;
fn solve_tridiagonal(dl: &[f64], d: &[f64], du: &[f64], y: ArrayView1<f64>) -> Array1<f64> {
    let n = d.len();
    debug_assert!(y.len() == n);
    debug_assert!(dl.len() == n - 1);
    debug_assert!(du.len() == n - 1);

    let mut scratch = Array1::<f64>::zeros(n);
    let mut x = y.to_owned();

    scratch[0] = du[0] / d[0];
    x[0] /= d[0];
    for ix in 1..n {
        if ix < n - 1 {
            scratch[ix] = du[ix] / (d[ix] - dl[ix - 1] * scratch[ix - 1]);
        }
        x[ix] = (x[ix] - dl[ix - 1] * x[ix - 1]) / (d[ix] - dl[ix - 1] * scratch[ix - 1]);
    }
    for ix in (0..(n - 1)).rev() {
        x[ix] -= scratch[ix] * x[ix + 1];
    }
    x
}
fn cubic_spline_3pts(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    t: Array1<isize>,
) -> (Array1<isize>, Array1<f64>) {
    let dx1 = x[1] - x[0];
    let dx2 = x[2] - x[1];
    let rdx1 = 1.0 / dx1;
    let rdx2 = 1.0 / dx2;
    let dy1 = y[1] - y[0];
    let dy2 = y[2] - y[1];

    let d = vec![2.0 * dx1, 2.0 * (rdx1 + rdx2), 2.0 * dx2];
    let dl = vec![rdx1, rdx2];
    let du = vec![rdx1, rdx2];

    let v1 = 3.0 * dy1 * rdx1 * rdx1;
    let v3 = 3.0 * dy2 * rdx2 * rdx2;
    let v = Array1::from_iter([v1, v1 + v3, v3]);

    // let s = mat.solve_tridiagonal(&v).unwrap();
    let s = solve_tridiagonal(&dl, &d, &du, v.view());
    let a1 = s[0] * dx1 - dy1;
    let b1 = -s[1] * dx1 + dy1;
    let a2 = s[1] * dx2 - dy2;
    let b2 = -s[2] * dx2 + dy2;
    let out = t
        .iter()
        .map(|tt| *tt as f64)
        .map(|tt| match find_ind(&get_cow_slice(&x), &tt) {
            1 => {
                let lam = (tt - x[0]) / dx1;
                let lamc = 1.0 - lam;
                lamc * y[0] + lam * y[1] + lam * lamc * (a1 * lamc + b1 * lam)
            }
            2 => {
                let lam = (tt - x[1]) / dx2;
                let lamc = 1.0 - lam;
                lamc * y[1] + lam * y[2] + lam * lamc * (a2 * lamc + b2 * lam)
            }
            3 => {
                if tt == x[2] {
                    y[2]
                } else {
                    panic!("Out of bounds")
                }
            }
            _ => panic!("Out of bounds"),
        })
        .collect();
    (t, out)

    // x.t.into_iter().map(|tt| match tt.cmp(x[1]) {});
}
fn cubic_spline_large(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    t: Array1<isize>,
) -> (Array1<isize>, Array1<f64>) {
    let n = x.len();
    let dx: Vec<_> = (1..n).map(|i| x[i] - x[i - 1]).collect();
    // let d2x: Vec<_> = (1..n - 1).map(|i| dx[i] - dx[i - 1]).collect();
    let slope: Vec<_> = (1..n).map(|i| (y[i] - y[i - 1]) / dx[i - 1]).collect();
    assert_eq!(dx.len(), n - 1);
    let mut d = vec![0.0f64; n];
    let mut dl = vec![0.0f64; n - 1];
    let mut du = vec![0.0f64; n - 1];
    let mut b = Array1::zeros(n);

    for i in 1..(n - 1) {
        d[i] = 2.0 * (dx[i] + dx[i - 1]);
        dl[i - 1] = dx[i];
        du[i] = dx[i - 1];
        b[i] = 3.0 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i]);
    }
    d[0] = dx[1];
    du[0] = x[2] - x[0];
    b[0] = ((dx[0] + 2.0 * du[0]) * dx[1] * slope[0] + dx[0] * dx[0] * slope[1]) / du[0];

    d[n - 1] = dx[n - 3];
    dl[n - 2] = x[n - 1] - x[n - 3];
    b[n - 1] = (dx[n - 2] * dx[n - 2] * slope[n - 3]
        + (2.0 * dl[n - 2] + dx[n - 2]) * dx[n - 3] * slope[n - 2])
        / dl[n - 2];

    let s = solve_tridiagonal(&dl, &d, &du, b.view());

    // dbg!(&s);
    let mut c = Array2::zeros((4, n - 1));
    for i in 0..n - 1 {
        let t = (s[i] + s[i + 1] - 2.0 * slope[i]) / dx[i];
        c[(0, i)] = t / dx[i];
        c[(1, i)] = (slope[i] - s[i]) / dx[i] - t;
        c[(2, i)] = s[i];
        c[(3, i)] = y[i];
    }
    // dbg!(&c);
    let x_slice = get_cow_slice(&x);
    let mut out = Array1::zeros(t.len());
    let mut ind = 0usize;
    // dbg!(&t);
    // dbg!(&x);
    for (j, tt) in t.iter().enumerate() {
        let tt = *tt as f64;

        ind = find_ind(&x_slice[ind.saturating_sub(1)..], &tt) + ind.saturating_sub(1);
        if ind == n {
            if tt == x[n - 1] {
                out[j] = y[n - 1];
            } else {
                panic!("Out of bounds")
            }
        } else if ind < n && ind > 0 {
            let lam = tt - x[ind - 1];
            let i = ind - 1;
            out[j] =
                c[(3, i)] + c[(2, i)] * lam + c[(1, i)] * lam * lam + c[(0, i)] * lam * lam * lam;
        } else {
            panic!("Out of bounds")
        }
    }
    (t, out)
}

pub fn cubic_spline_impl(
    n: usize,
    extrema_pos: ArrayView1<isize>,
    extrema_val: ArrayView1<f64>,
) -> (Array1<isize>, Array1<f64>) {
    let esize = extrema_pos.len();
    let t: Array1<_> = (0..n as isize)
        .filter(|tt| *tt >= extrema_pos[0] && *tt <= extrema_pos[esize - 1])
        .collect();
    if esize <= 3 {
        cubic_spline_3pts(extrema_pos.mapv(|x| x as f64).view(), extrema_val, t)
    } else {
        cubic_spline_large(extrema_pos.mapv(|x| x as f64).view(), extrema_val, t)
    }
}
// bisection is a faster algorithm  to look up indices in general
// but in our case, the index we need to find is (I think) either 0 or 1, so it is actually faster
// to just check these two. In fact bisect probably has worst case performance with our situation
// right?
fn _find_ind_bisect<T: PartialOrd<T>>(a: &[T], x: &T) -> usize {
    let mut lo = 0;
    let mut hi = a.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x < &a[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}
fn find_ind<T: PartialOrd<T>>(a: &[T], x: &T) -> usize {
    if x >= &a[a.len() - 1] {
        return a.len();
    }
    for (i, ai) in a.iter().enumerate() {
        if x < ai {
            return i;
        }
    }
    unreachable!();
}
#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_cubic_spline_3pts() {
        let x = Array1::from_iter([0.0, 4.0, 10.0]);
        let y = Array1::from_iter([1.0, -2.0, 4.0]);
        let t = Array1::from_iter(0isize..=10isize);
        let (_, out) = cubic_spline_3pts(x.view(), y.view(), t);
        dbg!(&out);
        assert_eq!(out[0], y[0]);
        assert_eq!(out[10], y[2]);
        assert_eq!(out[4], y[1]);
    }
    #[test]
    fn test_cubic_spline_large() {
        let x = Array1::from_iter([0.0, 4.0, 7.0, 10.0]);
        let y = Array1::from_iter([1.0, -2.0, 6.0, 4.0]);
        let t = Array1::from_iter(0isize..=10isize);
        let (_, out) = cubic_spline_large(x.view(), y.view(), t);
        dbg!(&out);
        assert_eq!(out[0], y[0]);
        assert_eq!(out[4], y[1]);
        assert_eq!(out[7], y[2]);
        assert_eq!(out[10], y[3]);
    }
    #[test]
    fn test_bisect() {
        assert_eq!(find_ind(&[0.0, 1.0, 2.0], &1.0), 2);
        assert_eq!(find_ind(&[0.0, 1.0, 2.0], &0.99), 1);
        assert_eq!(find_ind(&[0.0, 1.0, 2.0], &0.0), 1);
        assert_eq!(find_ind(&[0.0, 1.0, 2.0], &1.99), 2);
        assert_eq!(find_ind(&[0.0, 1.0, 2.0], &2.0), 3);
        assert_eq!(find_ind(&[0.0, 1.0, 2.0], &3.0), 3);
        assert_eq!(find_ind(&[0.0, 1.0, 2.0], &-2.0), 0);
    }
}
