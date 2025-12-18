use crate::ensemble::_eemd;
use crate::noise::normal_mt_impl;
use crate::{common::RsEMDOut, emd_impl::emd_impl};
use numpy::ndarray::{prelude::*, Dimension};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Mutex;
//ceemdan parameters
// const C_TRIALS: usize = 100;
const NOISE_SCALE: f64 = 1.0;
const C_RANGE_THRESH: f64 = 0.01;
const C_TOTAL_POWER_THRESH: f64 = 0.05;
const C_MAX_IMF: usize = 100;

fn make_noise_emd(seed: Option<u32>, trials: usize, n: usize, parallel: bool) -> Vec<RsEMDOut> {
    let all_noise = normal_mt_impl(seed, (n, trials), NOISE_SCALE);

    let noise_closure = |i| {
        let (imfs, resid) = emd_impl(all_noise.column(i), None);
        let imf_sd = imfs.row(0).std(0.0);
        (imfs / imf_sd, resid / imf_sd)
    };
    if parallel {
        (0..trials).into_par_iter().map(noise_closure).collect()
    } else {
        (0..trials).map(noise_closure).collect()
    }
}
// x += y*c without more allocations
fn add_to_vec(x: &mut ArrayViewMut1<f64>, y: ArrayView1<f64>, c: f64) {
    let n = x.len();
    for i in 0..n {
        x[i] += y[i] * c;
    }
}
trait Viewable<'a, D: Dimension> {
    fn views(&'a self) -> Vec<ArrayView<'a, f64, D>>;
}
impl<'a, D: Dimension> Viewable<'a, D> for Vec<Array<f64, D>> {
    fn views(&'a self) -> Vec<ArrayView<'a, f64, D>> {
        self.iter().map(|x| x.view()).collect()
    }
}
pub(crate) fn ceemdan_impl(
    val: ArrayView1<f64>,
    trials: usize,
    max_imf: Option<usize>,
    seed: Option<u32>,
    epsilon: f64,
    parallel: bool,
) -> RsEMDOut {
    let scale_s = val.std(0.0);
    let val = &val / scale_s;
    let n = val.len();
    // dbg!(trials);
    // let trials = C_TRIALS;

    // dbg!(&all_noise);

    let all_noise_emd: Vec<RsEMDOut> = make_noise_emd(seed, trials, n, parallel);
    // dbg!(&all_noise_emd);

    let noise_emd_1: Vec<_> = all_noise_emd.iter().map(|x| x.0.row(0)).collect();

    let mut all_cimfs = vec![_eemd(val.view(), trials, &noise_emd_1, epsilon, parallel)];
    // dbg!(&all_cimfs[0]);

    let mut prev_res = &val - &all_cimfs[0];
    // let mut scaled_residue = all_cimfs[0].clone();
    let mut scaled_residue = prev_res.clone();

    for i in 0..C_MAX_IMF {
        if c_end_condition(scaled_residue.view(), &all_cimfs.views(), max_imf, i) {
            break;
        }
        let beta = prev_res.std(0.0) * epsilon;

        let local_mean = Mutex::new(Array1::<f64>::zeros(n));
        let trial_closure = |trial| {
            let cur_noise_imf_resid: &RsEMDOut = &all_noise_emd[trial];
            let n_noise_imf = cur_noise_imf_resid.0.shape()[0];
            let mut res = prev_res.clone();

            if n_noise_imf == all_cimfs.len() {
                // res += &(&cur_noise_imf_resid.1 * beta);
                add_to_vec(&mut res.view_mut(), cur_noise_imf_resid.1.view(), beta);
            } else if n_noise_imf > all_cimfs.len() {
                // res += &(&cur_noise_imf_resid.0.row(all_cimfs.len()) * beta);
                add_to_vec(
                    &mut res.view_mut(),
                    cur_noise_imf_resid.0.row(all_cimfs.len()),
                    beta,
                );
            }
            let (_, resid) = emd_impl(res.view(), Some(1));
            // the threads need to lock the local mean array to update it. That's fine because it is
            // much quicker than calculating the emd
            let mut lm = local_mean.lock().unwrap();
            for i in 0..lm.len() {
                lm[i] += resid[i] / (trials as f64);
            }
        };
        if parallel {
            (0..trials).into_par_iter().for_each(trial_closure);
        } else {
            (0..trials).for_each(trial_closure);
        }
        let local_mean = local_mean.into_inner().unwrap();

        let imf = &prev_res - &local_mean;
        scaled_residue -= &imf;
        all_cimfs.push(imf);
        prev_res = local_mean.into_owned();
    }
    // dbg!(all_cimfs.len());
    let mut imf_arr = Array2::zeros((all_cimfs.len(), n));
    for (i, cur_imf) in all_cimfs.into_iter().enumerate() {
        let row = imf_arr.slice_mut(s![i, ..]);
        let scaled_imf = cur_imf * scale_s;
        (scaled_imf).move_into(row);
    }

    (imf_arr, &scaled_residue * scale_s)
}
fn c_end_condition(
    scaled_residue: ArrayView1<f64>,
    all_cimfs: &[ArrayView1<f64>],
    max_imf: Option<usize>,
    i: usize,
) -> bool {
    let n_imfs = all_cimfs.len();
    if max_imf.is_some_and(|mi| n_imfs >= mi) {
        return true;
    }
    let (emd, _) = emd_impl(scaled_residue, Some(1));
    if i == 0 {
        // dbg!(&scaled_residue);
        // dbg!(&emd);
    }
    if emd.shape()[0] == 0 {
        return true;
    }
    let residue_range = scaled_residue
        .iter()
        .copied()
        .reduce(f64::max)
        .unwrap_or(0.0)
        - scaled_residue
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(0.0);
    if i == 0 {
        // dbg!(residue_range);
    }
    if residue_range < C_RANGE_THRESH {
        return true;
    }
    let residue_power = scaled_residue.iter().map(|x| x.abs()).sum::<f64>();
    if i == 0 {
        // dbg!(residue_power);
    }
    if residue_power < C_TOTAL_POWER_THRESH {
        return true;
    }

    false
}
