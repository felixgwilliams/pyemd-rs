use anyhow::{bail, Error};
use numpy::{ndarray::prelude::*, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}
#[derive(Debug, Clone)]
#[pyclass]
struct FindExtremaOutput {
    max_pos: Vec<f64>,
    max_val: Vec<f64>,
    min_pos: Vec<f64>,
    min_val: Vec<f64>,
    zc_ind: Vec<usize>,
}
#[pymethods]
impl FindExtremaOutput {
    #[getter]
    fn max_pos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.max_pos.clone())
    }
    #[getter]
    fn max_val<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.max_val.clone())
    }
    #[getter]
    fn min_pos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.min_pos.clone())
    }
    #[getter]
    fn min_val<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.min_val.clone())
    }
    #[getter]
    fn zc_ind<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_vec(py, self.zc_ind.clone())
    }
}

fn find_extrema_simple_impl(
    pos: ArrayView1<f64>,
    val: ArrayView1<f64>,
) -> Result<FindExtremaOutput, Error> {
    let n = val.len();
    if pos.len() != n {
        bail!("val and pos must have the same length")
    }
    let zc = find_zero_crossing_impl(val.as_standard_layout().as_slice().unwrap());
    let (minpos, maxpos) = find_extrema_pos_impl(val.as_standard_layout().as_slice().unwrap());
    Ok(FindExtremaOutput {
        max_pos: maxpos.iter().map(|i| pos[*i]).collect(),
        min_pos: minpos.iter().map(|i| pos[*i]).collect(),
        max_val: maxpos.iter().map(|i| val[*i]).collect(),
        min_val: minpos.iter().map(|i| val[*i]).collect(),
        zc_ind: zc,
    })
}
const fn midpoint(a: usize, b: usize) -> usize {
    let sum = a + b;
    match sum % 4 {
        0 => (sum) / 2,
        2 => (sum) / 2,
        1 => (sum - 1) / 2,
        3 => sum.div_ceil(2),
        _ => unreachable!(),
    }
}
fn find_zero_crossing_impl(val: &[f64]) -> Vec<usize> {
    let n = val.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        if val[0] == 0.0 {
            return vec![0];
        } else {
            return vec![];
        }
    }

    let mut out = Vec::new();
    let mut debz = if val[0] == 0.0 { Some(0) } else { None };
    for i in 0..n - 1 {
        if val[i + 1] == 0.0 {
            if val[i] != 0.0 {
                debz = Some(i + 1);
            }
        } else if val[i] == 0.0 {
            out.push(midpoint(i, debz.unwrap()));
            debz = None;
        } else if val[i + 1].signum() != val[i].signum() {
            out.push(i)
        }
    }
    if let Some(debz) = debz {
        out.push(midpoint(debz, n - 1));
    }
    out
}

fn find_extrema_pos_impl(val: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let n = val.len();
    if n < 2 {
        return (vec![], vec![]);
    }
    let mut debs = None;
    let mut minout = Vec::new();
    let mut maxout = Vec::new();
    // let mut d = Vec::with_capacity(n - 1);
    // for i in 0..n - 1 {
    //     d.push(val[i + 1] - val[i]);
    // }
    for i in 0..n - 2 {
        let d1 = val[i + 2] - val[i + 1];
        let d2 = val[i + 1] - val[i]; // d[i]
        if d1 == 0.0 {
            debs.get_or_insert((i + 1, d2));
        } else if d2 == 0.0 {
        } else {
            if let Some((debs_inner, slope)) = debs {
                if slope > 0.0 && d2 < 0.0 {
                    maxout.push(midpoint(i, debs_inner));
                } else if slope < 0.0 && d2 > 0.0 {
                    minout.push(midpoint(i, debs_inner));
                }
            } else if d1.signum() != d2.signum() {
                if d2 < 0.0 {
                    minout.push(i + 1);
                }
                if d2 > 0.0 {
                    maxout.push(i + 1);
                }
            }
            debs = None;
        }
    }
    if let Some((debs_inner, slope)) = debs {
        if slope > 0.0 && val[n - 1] < val[n - 2] {
            maxout.push(midpoint(n - 2, debs_inner));
        } else if slope < 0.0 && val[n - 1] > val[n - 2] {
            minout.push(midpoint(n - 2, debs_inner));
        }
    }
    (minout, maxout)
}

#[pyfunction]
fn find_extrema_simple(
    py: Python,
    pos: PyReadonlyArray1<f64>,
    val: PyReadonlyArray1<f64>,
) -> PyResult<FindExtremaOutput> {
    let pos = pos.as_array();
    let val = val.as_array();

    let out = py.allow_threads(|| find_extrema_simple_impl(pos, val));
    // let out = find_extrema_simple_impl(val, pos);
    Ok(out?)
}

/// A Python module implemented in Rust.
#[pymodule]
fn _pyemd_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(find_extrema_simple, m)?)?;
    m.add_class::<FindExtremaOutput>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_zc() {
        assert_eq!(
            find_zero_crossing_impl(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vec![3]
        );
        assert_eq!(
            find_zero_crossing_impl(&[0.0, 1.0, 2.0, -3.0, 0.0]),
            vec![0, 2, 4]
        );
        assert_eq!(
            find_zero_crossing_impl(&[1.0, 2.0, -1.0, -2.0, 3.0, -1.0]),
            vec![1, 3, 4]
        );
        assert_eq!(find_zero_crossing_impl(&[1., 2., 0., 0., -1.]), vec![2]);
        assert_eq!(find_zero_crossing_impl(&[1., 2., 0., 0., 1.]), vec![2]);
        assert_eq!(find_zero_crossing_impl(&[1., 2., 0., 0., 0.]), vec![3]);
        assert_eq!(find_zero_crossing_impl(&[0., 0.]), vec![0]);
        assert_eq!(find_zero_crossing_impl(&[0., 1.]), vec![0]);
        assert_eq!(find_zero_crossing_impl(&[0., -1.]), vec![0]);
        assert_eq!(find_zero_crossing_impl(&[1., 0.]), vec![1]);
        assert_eq!(find_zero_crossing_impl(&[1., 1.]), Vec::<usize>::new());
        assert_eq!(find_zero_crossing_impl(&[1., -1.]), vec![0]);
        assert_eq!(find_zero_crossing_impl(&[-1., 0.]), vec![1]);
        assert_eq!(find_zero_crossing_impl(&[-1., 1.]), vec![0]);
        assert_eq!(find_zero_crossing_impl(&[-1., -1.]), Vec::<usize>::new());
        assert_eq!(find_zero_crossing_impl(&[0.]), vec![0]);
        assert_eq!(find_zero_crossing_impl(&[1.]), Vec::<usize>::new());
        assert_eq!(find_zero_crossing_impl(&[]), Vec::<usize>::new());
    }
    #[test]
    fn test_find_exrm() {
        assert_eq!(
            find_extrema_pos_impl(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (Vec::<usize>::new(), Vec::<usize>::new())
        );
        assert_eq!(
            find_extrema_pos_impl(&[0.0, 1.0, 2.0, -3.0, 0.0]),
            (vec![3], vec![2])
        );
        assert_eq!(
            find_extrema_pos_impl(&[1.0, 2.0, -1.0, -2.0, 3.0, -1.0]),
            (vec![3], vec![1, 4])
        );
        assert_eq!(
            find_extrema_pos_impl(&[1., 2., 0., 0., -1.]),
            (vec![], vec![1])
        );
        assert_eq!(
            find_extrema_pos_impl(&[1., 2., 0., 0., 1.]),
            (vec![2], vec![1]) // zerogap2
        );
        assert_eq!(
            find_extrema_pos_impl(&[1., 2., 0., 0., 0.]),
            (vec![], vec![1])
        );
        assert_eq!(find_extrema_pos_impl(&[0., 0.]), (vec![], vec![]));
        assert_eq!(find_extrema_pos_impl(&[0., 1.]), (vec![], vec![]));
        assert_eq!(find_extrema_pos_impl(&[0., -1.]), (vec![], vec![]));
        assert_eq!(find_extrema_pos_impl(&[1., 0.]), (vec![], vec![]));
        assert_eq!(find_extrema_pos_impl(&[1., 1.]), (vec![], vec![]));
        assert_eq!(find_extrema_pos_impl(&[1., -1.]), (vec![], vec![]));
        assert_eq!(find_extrema_pos_impl(&[-1., 0.]), (vec![], vec![]));
        assert_eq!(find_extrema_pos_impl(&[-1., 1.]), (vec![], vec![]));
        assert_eq!(find_extrema_pos_impl(&[-1., -1.]), (vec![], vec![]));
        assert_eq!(find_extrema_pos_impl(&[0.]), (vec![], vec![]));
        assert_eq!(find_extrema_pos_impl(&[1.]), (vec![], vec![]));
        assert_eq!(find_extrema_pos_impl(&[]), (vec![], vec![]));

        assert_eq!(
            find_extrema_pos_impl(&[-1., 0., 1., 0., -1., 0., 3., 0., -9., 0.]),
            (vec![4, 8], vec![2, 6])
        );
        assert_eq!(
            find_extrema_pos_impl(&[-1., 0., 1., 1., 0., -1., 0., 3., 0., -9., 0.]),
            (vec![5, 9], vec![2, 7])
        )
    }
}
