use pyo3::IntoPyObject;

// let out = find_extrema_simple_impl(val, pos);
pub const MAX_ITERATION: usize = 1000;

// // EMD parameters
pub const SVAR_THRESH: f64 = 0.001;
pub const ENERGY_RATIO_THRESH: f64 = 0.2;
pub const STD_THRESH: f64 = 0.2;
pub const RANGE_THRESH: f64 = 0.001;
pub const TOTAL_POWER_THRESH: f64 = 0.005;
pub const NOISE_SCALE: f64 = 1.0;
// CEEMDAN Parameters
pub const C_RANGE_THRESH: f64 = 0.01;
pub const C_TOTAL_POWER_THRESH: f64 = 0.05;
pub const C_MAX_IMF: usize = 100;
#[derive(Debug, Clone, IntoPyObject)]
pub(crate) struct EmdOpts {
    pub max_iteration: usize,
    pub svar_thresh: f64,
    pub energy_ratio_thresh: f64,
    pub std_thresh: f64,
    pub range_thresh: f64,
    pub total_power_thresh: f64,
}
impl Default for EmdOpts {
    fn default() -> Self {
        Self {
            max_iteration: MAX_ITERATION,
            svar_thresh: SVAR_THRESH,
            energy_ratio_thresh: ENERGY_RATIO_THRESH,
            std_thresh: STD_THRESH,
            range_thresh: RANGE_THRESH,
            total_power_thresh: TOTAL_POWER_THRESH,
        }
    }
}

#[derive(Debug, Clone, IntoPyObject)]
pub(crate) struct CeemdanOpts {
    pub noise_scale: f64,
    pub c_range_thresh: f64,
    pub c_total_power_thresh: f64,
    pub c_max_imf: usize,
}
impl Default for CeemdanOpts {
    fn default() -> Self {
        Self {
            noise_scale: NOISE_SCALE,
            c_range_thresh: C_RANGE_THRESH,
            c_total_power_thresh: C_TOTAL_POWER_THRESH,
            c_max_imf: C_MAX_IMF,
        }
    }
}
