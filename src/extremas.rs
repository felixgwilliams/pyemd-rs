use numpy::{ndarray::prelude::*, PyArray1};
use pyo3::prelude::*;
// use std::collections::BinaryHeap;

use crate::common::get_cow_slice;
pub fn find_extrema_pos_impl(val: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let n = val.len();
    if n < 2 {
        return (vec![], vec![]);
    }
    let mut minout = Vec::new();
    let mut maxout = Vec::new();
    let mut level = Vec::new();
    let mut level_ends = Vec::new();
    let mut cur_level = None;

    if val[1] == val[0] {
        level.push(0);
        cur_level = Some(0);
    }

    for (i, window) in val.windows(3).enumerate() {
        let (prev, curr, next) = (window[0], window[1], window[2]);
        let d2 = curr - prev;
        let d1 = next - curr;
        let d1_pos = d1 > 0.0;
        let d2_pos = d2 > 0.0;
        if d1 == 0.0 {
            if cur_level.is_none() {
                level.push(i + 1);
                cur_level.get_or_insert(i + 1);
            }
        } else {
            if cur_level.is_some() {
                level_ends.push(i + 1);
                cur_level = None;
            }

            if d2 != 0.0 && d1_pos ^ d2_pos {
                // if d2 != 0.0 && d1.signum() != d2.signum() {
                if d2_pos {
                    maxout.push(i + 1);
                } else {
                    minout.push(i + 1);
                }
            }
        }
    }
    // if we are in a "level" run at the end, we remove the entry
    if cur_level.is_some() {
        level.pop();
        // level_ends.push(n - 1);
    }
    // dbg!((&level, &level_ends, &cur_level,));
    assert!(level.len() == level_ends.len());
    // if there are no duplicates, we can return
    if level.is_empty() {
        return (minout, maxout);
    }
    // We need to do a second pass
    // It may be faster to append to the vectors then sort again. We can test this later!
    // let mut minout = BinaryHeap::from(minout);
    // let mut maxout = BinaryHeap::from(maxout);
    for (start, end) in level.iter().copied().zip(level_ends) {
        if start == 1 {
            continue;
        }
        let in_slope = if start == 0 {
            val[n - 1] - val[n - 2]
        } else {
            val[start] - val[start - 1]
        };
        let out_slope = val[end + 1] - val[end - 1];
        match (in_slope > 0.0, out_slope < 0.0) {
            (true, true) => maxout.push(midpoint(start, end)),
            (false, false) if in_slope < 0.0 => minout.push(midpoint(start, end)),
            _ => {}
        }
    }
    minout.sort_unstable();
    maxout.sort_unstable();
    (minout, maxout)
    // (minout.into_sorted_vec(), maxout.into_sorted_vec())
}
pub fn find_zero_crossing_impl(val: &[f64]) -> Vec<usize> {
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
    for (i, window) in val.windows(2).enumerate() {
        let (curr, next) = (window[0], window[1]);
        let curr_is_zero = curr == 0.0;
        let next_is_zero = next == 0.0;
        match (curr_is_zero, next_is_zero) {
            (false, true) => debz = Some(i + 1),
            (true, false) => {
                out.push(midpoint(i, debz.unwrap()));
                debz = None;
            }
            (false, false) if (curr > 0.0) ^ (next > 0.0) => {
                out.push(i);
            }
            _ => {}
        }
    }
    if let Some(debz) = debz {
        out.push(midpoint(debz, n - 1));
    }
    out
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
pub fn fill_by_index<T: Copy>(pos: &[usize], val: &[T]) -> Vec<T> {
    let mut max_val = Vec::with_capacity(pos.len());
    pos.iter().for_each(|j| max_val.push(val[*j]));
    max_val
}
pub fn find_extrema_simple_impl(val: ArrayView1<f64>) -> FindExtremaOutput {
    let val_slice = get_cow_slice(&val);
    let zc = find_zero_crossing_impl(&val_slice);
    let (minpos, maxpos) = find_extrema_pos_impl(&val_slice);
    let max_val = fill_by_index(&maxpos, &val_slice);
    let min_val = fill_by_index(&minpos, &val_slice);
    FindExtremaOutput {
        max_val,
        min_val,
        max_pos: maxpos,
        min_pos: minpos,
        zc_ind: zc,
    }
}
#[derive(Debug, Clone)]
#[pyclass]
pub struct FindExtremaOutput {
    pub max_pos: Vec<usize>,
    pub max_val: Vec<f64>,
    pub min_pos: Vec<usize>,
    pub min_val: Vec<f64>,
    pub zc_ind: Vec<usize>,
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
impl FindExtremaOutput {
    pub fn get_lengths(&self) -> ExtremaLengths {
        ExtremaLengths {
            num_max: self.max_pos.len(),
            num_min: self.min_pos.len(),
            num_zc: self.zc_ind.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExtremaLengths {
    pub num_max: usize,
    pub num_min: usize,
    pub num_zc: usize,
}
impl Default for ExtremaLengths {
    fn default() -> Self {
        ExtremaLengths {
            num_max: usize::MAX,
            num_min: usize::MAX,
            num_zc: usize::MAX,
        }
    }
}
impl ExtremaLengths {
    pub fn num_extrema(&self) -> usize {
        self.num_max + self.num_min
    }
    pub fn imf_condition(&self) -> bool {
        let ext_no = self.num_extrema();
        ext_no.abs_diff(self.num_zc) < 2
    }
    pub fn diff(&self, other: &Self) -> usize {
        self.num_max.abs_diff(other.num_max)
            + self.num_min.abs_diff(other.num_min)
            + self.num_zc.abs_diff(other.num_zc)
    }
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
        );
        assert_eq!(
            find_extrema_pos_impl(&[
                52., 20., 75., 56., 65., 65., 37., 79., 73., 66., 9., 48., 57., 44., 75., 3., 34.,
                36., 38., 73.
            ]),
            (vec![1, 3, 6, 10, 13, 15], vec![2, 4, 7, 12, 14])
        )
    }
}
