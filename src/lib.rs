use numpy::{PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
mod ceemdan_impl;
mod common;
mod emd_impl;
mod ensemble;
mod extremas;
mod noise;
mod splines;

#[pyfunction]
fn find_extrema_simple(py: Python, val: PyReadonlyArray1<f64>) -> extremas::FindExtremaOutput {
    let val = val.as_array();

    // let out = find_extrema_simple_impl(val, pos);
    py.detach(|| extremas::find_extrema_simple_impl(val))
}

#[pyfunction]
fn find_extrema_simple_pos<'py>(
    py: Python<'py>,
    val: PyReadonlyArray1<'py, f64>,
) -> (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>) {
    let val = val.as_array();

    let (minpos, maxpos) =
        py.detach(|| extremas::find_extrema_pos_impl(val.as_standard_layout().as_slice().unwrap()));
    // let out = find_extrema_simple_impl(val, pos);
    (
        PyArray1::from_vec(py, minpos),
        PyArray1::from_vec(py, maxpos),
    )
}

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
    let (min_extrema_pos, min_extrema_val, max_extrema_pos, max_extrema_val) = py.detach(|| {
        emd_impl::prepare_points_simple_impl(
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

type SplineReturn<'py> = (Bound<'py, PyArray1<isize>>, Bound<'py, PyArray1<f64>>);
#[pyfunction]
fn cubic_spline<'py>(
    py: Python<'py>,
    n: usize,
    extrema_pos: PyReadonlyArray1<'py, isize>,
    extrema_val: PyReadonlyArray1<'py, f64>,
) -> PyResult<SplineReturn<'py>> {
    let extrema_pos = extrema_pos.as_array();
    let extrema_val = extrema_val.as_array();
    let (pos, interp) = py.detach(|| splines::cubic_spline_impl(n, extrema_pos, extrema_val));
    Ok((pos.to_pyarray(py), interp.to_pyarray(py)))
}

type PyEMDOut<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);
#[pyfunction]
#[pyo3(signature = (val, max_imf=None))]
fn emd<'py>(
    py: Python<'py>,
    val: PyReadonlyArray1<'py, f64>,
    max_imf: Option<usize>,
) -> PyEMDOut<'py> {
    let val = val.as_array();

    // let out = find_extrema_simple_impl(val, pos);
    let (imfs, resid) = py.detach(|| emd_impl::emd_impl(val, max_imf));
    (imfs.to_pyarray(py), resid.to_pyarray(py))
}

#[pyfunction]
fn normal_mt(
    py: Python<'_>,
    seed: Option<u32>,
    size: usize,
    scale: f64,
) -> Bound<'_, PyArray1<f64>> {
    let arr = py.detach(|| noise::normal_mt_impl(seed, size, scale));
    arr.to_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (val, trials=100, max_imf=None, seed=None, epsilon=0.005, *, parallel=true))]
fn ceemdan<'py>(
    py: Python<'py>,
    val: PyReadonlyArray1<'py, f64>,
    trials: usize,
    max_imf: Option<usize>,
    seed: Option<u32>,
    epsilon: f64,
    parallel: bool,
) -> PyEMDOut<'py> {
    // PyEMDOut<'py>
    let val = val.as_array();
    // py.detach(|| ceemdan_impl(val, max_imf));
    let (imfs, resid) =
        py.detach(|| ceemdan_impl::ceemdan_impl(val, trials, max_imf, seed, epsilon, parallel));
    (imfs.to_pyarray(py), resid.to_pyarray(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn _pyemd_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_extrema_simple, m)?)?;
    m.add_function(wrap_pyfunction!(find_extrema_simple_pos, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_points_simple, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_spline, m)?)?;
    m.add_function(wrap_pyfunction!(emd, m)?)?;
    m.add_function(wrap_pyfunction!(normal_mt, m)?)?;
    m.add_function(wrap_pyfunction!(ceemdan, m)?)?;
    m.add_class::<extremas::FindExtremaOutput>()?;
    Ok(())
}
