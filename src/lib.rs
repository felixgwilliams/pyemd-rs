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
    max_pos: Vec<usize>,
    max_val: Vec<f64>,
    min_pos: Vec<usize>,
    min_val: Vec<f64>,
    zc_ind: Vec<usize>,
}
#[pymethods]
impl FindExtremaOutput {
    #[getter]
    fn max_pos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_vec(py, self.max_pos.clone())
    }
    #[getter]
    fn max_val<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.max_val.clone())
    }
    #[getter]
    fn min_pos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
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

fn find_extrema_simple_impl(val: ArrayView1<f64>) -> FindExtremaOutput {
    let zc = find_zero_crossing_impl(val.as_standard_layout().as_slice().unwrap());
    let (minpos, maxpos) = find_extrema_pos_impl(val.as_standard_layout().as_slice().unwrap());
    FindExtremaOutput {
        max_val: maxpos.iter().map(|i| val[*i]).collect(),
        min_val: minpos.iter().map(|i| val[*i]).collect(),
        max_pos: maxpos,
        min_pos: minpos,
        zc_ind: zc,
    }
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
fn find_extrema_simple(py: Python, val: PyReadonlyArray1<f64>) -> FindExtremaOutput {
    let val = val.as_array();

    // let out = find_extrema_simple_impl(val, pos);
    py.allow_threads(|| find_extrema_simple_impl(val))
}

#[pyfunction]
fn find_extrema_simple_pos<'py>(
    py: Python<'py>,
    val: PyReadonlyArray1<'py, f64>,
) -> (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>) {
    let val = val.as_array();

    let (minpos, maxpos) =
        py.allow_threads(|| find_extrema_pos_impl(val.as_standard_layout().as_slice().unwrap()));
    // let out = find_extrema_simple_impl(val, pos);
    (
        PyArray1::from_vec(py, minpos),
        PyArray1::from_vec(py, maxpos),
    )
}

fn prepare_points_simple_impl(
    val: &[f64],
    min_pos: &[usize],
    max_pos: &[usize],
    nbsym: usize,
) -> (Vec<isize>, Vec<f64>, Vec<isize>, Vec<f64>) {
    let end_min = min_pos.len();
    let end_max = max_pos.len();
    let n = val.len();

    let (mut lmax, mut lmin, lsym): (Vec<_>, Vec<_>, usize) = if max_pos[0] < min_pos[0] {
        if val[0] > val[min_pos[0]] {
            // dbg!("l1");
            (
                max_pos[1..end_max.min(nbsym + 1)]
                    .iter()
                    .copied()
                    .rev()
                    .collect(),
                min_pos[0..end_min.min(nbsym)]
                    .iter()
                    .copied()
                    .rev()
                    .collect(),
                max_pos[0],
            )
        } else {
            // dbg!("l2");
            (
                max_pos[0..end_max.min(nbsym)]
                    .iter()
                    .copied()
                    .rev()
                    .collect(),
                min_pos[0..end_min.min(nbsym - 1)]
                    .iter()
                    .copied()
                    .rev()
                    .chain([0].iter().copied())
                    .collect(),
                0,
            )
        }
    } else if val[0] < val[max_pos[0]] {
        // dbg!("l3");
        (
            max_pos[0..end_max.min(nbsym)]
                .iter()
                .copied()
                .rev()
                .collect(),
            min_pos[1..end_min.min(nbsym + 1)]
                .iter()
                .copied()
                .rev()
                .collect(),
            min_pos[0],
        )
    } else {
        // dbg!("l4");
        (
            max_pos[0..end_max.min(nbsym - 1)]
                .iter()
                .copied()
                .rev()
                .chain([0].iter().copied())
                .collect(),
            min_pos[0..end_min.min(nbsym)]
                .iter()
                .copied()
                .rev()
                .collect(),
            0,
        )
    };
    let end_chain = &[n - 1];
    let (mut rmax, mut rmin, rsym): (Vec<_>, Vec<_>, usize) =
        if max_pos[end_max - 1] < min_pos[end_min - 1] {
            if val[n - 1] < val[max_pos[end_max - 1]] {
                // dbg!("r1");
                (
                    max_pos[end_max.saturating_sub(nbsym)..]
                        .iter()
                        .copied()
                        .rev()
                        .collect(),
                    min_pos[end_min.saturating_sub(nbsym + 1)..end_min - 1]
                        .iter()
                        .copied()
                        .rev()
                        .collect(),
                    min_pos[end_min - 1],
                )
            } else {
                // dbg!("r2");
                (
                    max_pos[(end_max + 1).saturating_sub(nbsym)..]
                        .iter()
                        .copied()
                        .chain(end_chain.iter().copied())
                        .rev()
                        .collect(),
                    min_pos[end_min.saturating_sub(nbsym)..]
                        .iter()
                        .copied()
                        .rev()
                        .collect(),
                    n - 1,
                )
            }
        } else if val[n - 1] > val[min_pos[end_min - 1]] {
            // dbg!("r3");
            (
                max_pos[end_max.saturating_sub(nbsym + 1)..end_max - 1]
                    .iter()
                    .copied()
                    .rev()
                    .collect(),
                min_pos[end_min.saturating_sub(nbsym)..]
                    .iter()
                    .copied()
                    .rev()
                    .collect(),
                max_pos[end_max - 1],
            )
        } else {
            // dbg!("r4");
            (
                max_pos[end_max.saturating_sub(nbsym)..]
                    .iter()
                    .copied()
                    .rev()
                    .collect(),
                min_pos[(end_min + 1).saturating_sub(nbsym)..]
                    .iter()
                    .copied()
                    .chain(end_chain.iter().copied())
                    .rev()
                    .collect(),
                n - 1,
            )
        };
    if lmin.is_empty() {
        lmin = min_pos.to_owned();
    }
    if rmin.is_empty() {
        rmin = min_pos.to_owned();
    }
    if lmax.is_empty() {
        lmax = max_pos.to_owned();
    }
    if lmax.is_empty() {
        rmax = max_pos.to_owned();
    }

    let mut tlmin: Vec<_> = lmin
        .iter()
        .map(|x| 2 * lsym as isize - *x as isize)
        .collect();
    let mut tlmax: Vec<_> = lmax
        .iter()
        .map(|x| 2 * lsym as isize - *x as isize)
        .collect();
    if tlmin[0] > 0 || tlmax[0] > 0 {
        if lsym == 0 {
            panic!("Left edge bug")
        }
        if lsym == max_pos[0] {
            // dbg!("ml1");
            lmax = max_pos[0..end_max.min(nbsym)]
                .iter()
                .copied()
                .rev()
                .collect();
            tlmax = lmax
                .iter()
                .map(|x| 2 * lsym as isize - *x as isize)
                .collect();
        } else {
            // dbg!("ml2");
            lmin = min_pos[0..end_min.min(nbsym)]
                .iter()
                .copied()
                .rev()
                .collect();
            tlmin = lmin
                .iter()
                .map(|x| 2 * lsym as isize - *x as isize)
                .collect();
        }
        // lsym = 0
    }

    let mut trmin: Vec<_> = rmin
        .iter()
        .map(|x| 2 * rsym as isize - *x as isize)
        .collect();
    let mut trmax: Vec<_> = rmax
        .iter()
        .map(|x| 2 * rsym as isize - *x as isize)
        .collect();
    dbg!(&rmin);
    dbg!(&rmax);
    dbg!(rsym);
    dbg!(&trmin);
    dbg!(&trmax);
    dbg!(n);
    if trmin[trmin.len() - 1] < n as isize - 1 || trmax[trmax.len() - 1] < n as isize - 1 {
        if rsym == n - 1 {
            panic!("Right edge bug.")
        }
        if rsym == max_pos[end_max - 1] {
            dbg!("mr1");
            rmax = max_pos[end_max.saturating_sub(nbsym)..]
                .iter()
                .copied()
                .rev()
                .collect();
            trmax = rmax
                .iter()
                .map(|x| 2 * rsym as isize - *x as isize)
                .collect();
        } else {
            dbg!("mr2");
            rmin = min_pos[end_min.saturating_sub(nbsym)..]
                .iter()
                .copied()
                .rev()
                .collect();
            trmin = rmin
                .iter()
                .map(|x| 2 * rsym as isize - *x as isize)
                .collect();
        }
        // rsym = n - 1
    }
    let tmin_it: Vec<_> = tlmin
        .iter()
        .copied()
        .chain(min_pos.iter().map(|i| *i as isize))
        .chain(trmin.iter().copied())
        .collect();
    let tmax_it: Vec<_> = tlmax
        .iter()
        .copied()
        .chain(max_pos.iter().map(|i| *i as isize))
        .chain(trmax.iter().copied())
        .collect();
    let zmin_it: Vec<_> = lmin
        .iter()
        .map(|i| val[*i])
        .chain(min_pos.iter().map(|i| val[*i]))
        .chain(rmin.iter().map(|i| val[*i]))
        .collect();
    let zmax_it: Vec<_> = lmax
        .iter()
        .map(|i| val[*i])
        .chain(max_pos.iter().map(|i| val[*i]))
        .chain(rmax.iter().map(|i| val[*i]))
        .collect();

    let mut tmin = Vec::new();
    let mut zmin = Vec::new();
    for i in 0..tmin_it.len() - 1 {
        if tmin_it[i] != tmin_it[i + 1] {
            tmin.push(tmin_it[i]);
            zmin.push(zmin_it[i]);
        }
    }
    tmin.push(tmin_it[tmin_it.len() - 1]);
    zmin.push(zmin_it[tmin_it.len() - 1]);

    let mut tmax = Vec::new();
    let mut zmax = Vec::new();
    for i in 0..tmax_it.len() - 1 {
        if tmax_it[i] != tmax_it[i + 1] {
            tmax.push(tmax_it[i]);
            zmax.push(zmax_it[i]);
        }
    }
    tmax.push(tmax_it[tmax_it.len() - 1]);
    zmax.push(zmax_it[tmax_it.len() - 1]);

    // let (tmin, zmin): (Vec<_>, Vec<_>) = tmin_it
    //     .zip(zmin_it)
    //     .filter(|(t, _z)| minset.insert(*t))
    //     .unzip();
    // let (tmax, zmax): (Vec<_>, Vec<_>) = tmax_it
    //     .zip(zmax_it)
    //     .filter(|(t, _z)| maxset.insert(*t))
    //     .unzip();

    (tmin, zmin, tmax, zmax)
}

// trait SignedIndex {
//     type IndOutput;
//     fn getsi(&self, i: isize) -> Self::IndOutput;
// }

// impl<T: Copy> SignedIndex for &[T] {
//     type IndOutput = T;
//     fn getsi(&self, i: isize) -> Self::IndOutput {
//         let n = self.len();
//         let ind = if i >= 0 {
//             i as usize
//         } else {
//             n - (-i as usize)
//         };
//         self[ind]
//     }
// }

type PreparePointsOut<'py> = (
    Bound<'py, PyArray1<isize>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<isize>>,
    Bound<'py, PyArray1<f64>>,
);
#[pyfunction]
fn prepare_points_simple<'py>(
    py: Python<'py>,
    val: PyReadonlyArray1<'py, f64>,
    min_pos: PyReadonlyArray1<'py, usize>,
    max_pos: PyReadonlyArray1<'py, usize>,
    nsymb: usize,
) -> PyResult<PreparePointsOut<'py>> {
    let val = val.as_array();
    let min_pos = min_pos.as_array();
    let max_pos = max_pos.as_array();
    let (min_extrema_pos, min_extrema_val, max_extrema_pos, max_extrema_val) =
        py.allow_threads(|| {
            prepare_points_simple_impl(
                val.as_standard_layout().as_slice().unwrap(),
                min_pos.as_standard_layout().as_slice().unwrap(),
                max_pos.as_standard_layout().as_slice().unwrap(),
                nsymb,
            )
        });
    Ok((
        PyArray1::from_vec(py, min_extrema_pos),
        PyArray1::from_vec(py, min_extrema_val),
        PyArray1::from_vec(py, max_extrema_pos),
        PyArray1::from_vec(py, max_extrema_val),
    ))
}

/// A Python module implemented in Rust.
#[pymodule]
fn _pyemd_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(find_extrema_simple, m)?)?;
    m.add_function(wrap_pyfunction!(find_extrema_simple_pos, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_points_simple, m)?)?;
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
