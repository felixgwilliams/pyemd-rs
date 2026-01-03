use std::sync::Mutex;

use numpy::ndarray::Array1;
use numpy::ndarray::ArrayView1;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::emd_impl::emd_impl;
use crate::options::EmdOpts;

pub(crate) fn _eemd(
    val: ArrayView1<f64>,
    trials: usize,
    noise_emd_1: &[ArrayView1<f64>],
    epsilon: f64,
    parallel: bool,
    emd_opts: &EmdOpts,
) -> Array1<f64> {
    let n = val.len();

    let out = Mutex::new(Array1::zeros(n));
    let noise_closure = |t| {
        let val_noise: Array1<f64> = &val + (epsilon * &noise_emd_1[t]);
        let cur_emd = emd_impl(val_noise.view(), Some(1), emd_opts);
        let mut out = out.lock().unwrap();
        for i in 0..n {
            out[i] += cur_emd.0.row(0)[i] / (trials as f64);
        }
    };
    if parallel {
        (0..trials).into_par_iter().for_each(noise_closure);
    } else {
        (0..trials).for_each(noise_closure);
    }

    out.into_inner().unwrap()
}
