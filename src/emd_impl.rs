use numpy::ndarray::prelude::*;

use crate::{
    common::{get_cow_slice, RsEMDOut},
    extremas::find_extrema_simple_impl,
    options::EmdOpts,
    splines::cubic_spline_impl,
};

fn check_imf(
    imf: ArrayView1<f64>,
    imf_old: ArrayView1<f64>,
    zmin: ArrayView1<f64>,
    zmax: ArrayView1<f64>,
    emd_opts: &EmdOpts,
) -> bool {
    if zmin.iter().any(|z| *z > 0.0) || zmax.iter().any(|z| *z < 0.0) {
        return false;
    }
    if imf.map(|x| x * x).sum() < 1e-10 {
        return false;
    }
    let imf_diff = &imf - &imf_old;
    let imf_diff_sqsum = imf_diff.map(|x| x * x).sum();
    let svar = imf_diff_sqsum
        / (imf.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
            - imf.iter().min_by(|a, b| a.total_cmp(b)).unwrap());
    if svar < emd_opts.svar_thresh {
        return true;
    }
    let std: f64 = imf_diff
        .iter()
        .zip(imf)
        .map(|(x, y)| *x * *x / *y / *y)
        .sum();
    if std < emd_opts.std_thresh {
        return true;
    }
    let energy_ratio = imf_diff_sqsum / imf_old.map(|x| x * x).sum();
    if energy_ratio < emd_opts.energy_ratio_thresh {
        return true;
    }
    false
}
fn end_condition(resid: &ArrayView1<f64>, emd_opts: &EmdOpts) -> bool {
    let resmax = resid.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let resmin = resid.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let ressum: f64 = resid.iter().map(|x| x.abs()).sum();

    resmax - resmin < emd_opts.range_thresh || ressum < emd_opts.total_power_thresh
}
pub fn emd_impl(val: ArrayView1<f64>, max_imf: Option<usize>, emd_opts: &EmdOpts) -> RsEMDOut {
    let mut finished = false;
    let mut resid = val.to_owned();
    let n = val.len();
    let mut imfs = Vec::new();
    let mut imf_is_residual = false;
    '_all_imf: while !finished {
        let mut imf = resid.to_owned();
        let mut s_counter = 0;

        'cur_imf: for _i in 1..emd_opts.max_iteration {
            let imf_view = imf.view();

            let extremas = find_extrema_simple_impl(imf_view);
            let extremas_lengths = extremas.get_lengths();

            let ext_no = extremas_lengths.num_extrema();

            // let min_pos = Array1::from_iter(extremas.min_pos.iter().map(|x| *x as isize));
            // let max_pos = Array1::from_iter(extremas.max_pos.iter().map(|x| *x as isize));
            // let min_val = Array1::from_vec(extremas.min_val);
            // let max_val = Array1::from_vec(extremas.max_val);

            if ext_no > 2 {
                // dbg!(&extremas.min_pos);
                // dbg!(&extremas.max_pos);
                let (tmin, zmin, tmax, zmax) = prepare_points_simple_impl(
                    &get_cow_slice(&imf_view),
                    &extremas.min_pos,
                    &extremas.max_pos,
                    2,
                );
                let tmin = Array1::from_vec(tmin);
                let zmin = Array1::from_vec(zmin);
                let tmax = Array1::from_vec(tmax);
                let zmax = Array1::from_vec(zmax);
                // dbg!((n, &tmin, &tmax, &zmin, &zmax));
                let (_, min_spline) = cubic_spline_impl(n, tmin.view(), zmin.view());
                let (_, max_spline) = cubic_spline_impl(n, tmax.view(), zmax.view());
                let imf_old = imf.to_owned();
                // dbg!(&min_spline);

                // dbg!((imf.shape(), min_spline.shape(), max_spline.shape()));
                imf = &imf - (&min_spline + &max_spline) * 0.5;
                let imf_view = imf.view();

                let extremas2 = find_extrema_simple_impl(imf_view);
                let extremas_lengths2 = extremas2.get_lengths();
                if true {
                    // TODO: s_number active
                    if true {
                        //  TODO:check extrema diff
                        s_counter += 1;
                        if s_counter >= 0 //  TODO:replace with proper s_number check
                            && extremas_lengths2.imf_condition()
                            && check_imf(
                                imf_view,
                                imf_old.view(),
                                zmin.view(),
                                zmax.view(),
                                emd_opts,
                            )
                        {
                            break 'cur_imf;
                        }
                    } else {
                        s_counter = 0;
                    }
                }
            } else {
                finished = true;
                imf_is_residual = true;
                // break inner loop
                break 'cur_imf;
            }
        }
        resid -= &imf;
        imfs.push(imf);
        if max_imf.is_some_and(|m| m <= imfs.len()) || end_condition(&resid.view(), emd_opts) {
            // finished = true;
            break '_all_imf;
        }
        // dbg!(&imfs);
    }
    if imf_is_residual {
        let last_imf = imfs.pop().unwrap();
        resid = resid + last_imf;
    }
    let mut imf_arr = Array2::zeros((imfs.len(), n));
    for (i, cur_imf) in imfs.into_iter().enumerate() {
        let row = imf_arr.slice_mut(s![i, ..]);
        cur_imf.move_into(row);
    }

    (imf_arr, resid)
}

fn simple_get_l(
    val: &[f64],
    min_pos: &[usize],
    max_pos: &[usize],
    nbsym: usize,
    end_min: usize,
    end_max: usize,
) -> (Vec<usize>, Vec<usize>, usize) {
    if max_pos[0] < min_pos[0] {
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
    }
}

fn simple_get_r(
    val: &[f64],
    min_pos: &[usize],
    max_pos: &[usize],
    nbsym: usize,
    end_min: usize,
    end_max: usize,
) -> (Vec<usize>, Vec<usize>, usize) {
    let n = val.len();
    let end_chain = &[n - 1];
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
    }
}
pub fn prepare_points_simple_impl(
    val: &[f64],
    min_pos: &[usize],
    max_pos: &[usize],
    nbsym: usize,
) -> (Vec<isize>, Vec<f64>, Vec<isize>, Vec<f64>) {
    let end_min = min_pos.len();
    let end_max = max_pos.len();
    let n = val.len();

    let (mut lmax, mut lmin, lsym): (Vec<_>, Vec<_>, usize) =
        simple_get_l(val, min_pos, max_pos, nbsym, end_min, end_max);

    let (mut rmax, mut rmin, rsym) = simple_get_r(val, min_pos, max_pos, nbsym, end_min, end_max);
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
        } else {
            // dbg!("ml2");
            lmin = min_pos[0..end_min.min(nbsym)]
                .iter()
                .copied()
                .rev()
                .collect();
        }
        tlmax = lmax.iter().map(|x| -(*x as isize)).collect();
        tlmin = lmin.iter().map(|x| -(*x as isize)).collect();
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

    if trmin[trmin.len() - 1] < n as isize - 1 || trmax[trmax.len() - 1] < n as isize - 1 {
        if rsym == n - 1 {
            panic!("Right edge bug.")
        }
        if rsym == max_pos[end_max - 1] {
            // dbg!("mr1");
            rmax = max_pos[end_max.saturating_sub(nbsym)..]
                .iter()
                .copied()
                .rev()
                .collect();
        } else {
            // dbg!("mr2");
            rmin = min_pos[end_min.saturating_sub(nbsym)..]
                .iter()
                .copied()
                .rev()
                .collect();
        }
        trmax = rmax
            .iter()
            .map(|x| 2 * (n - 1) as isize - *x as isize)
            .collect();
        trmin = rmin
            .iter()
            .map(|x| 2 * (n - 1) as isize - *x as isize)
            .collect();
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

    (tmin, zmin, tmax, zmax)
}
