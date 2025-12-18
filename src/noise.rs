use numpy::ndarray::{prelude::*, Dimension, Shape};
use rand::RngCore;
use rand_mt::Mt;
fn get_random_seed() -> u32 {
    getrandom::u32().expect("Failed to generate seed")
}
struct DoubleMt {
    mt: Mt,
    cached_gauss: Option<f64>,
}
trait CachedGauss {
    fn push_gauss(&mut self, val: f64) -> bool;
    fn pop_gauss(&mut self) -> Option<f64>;
}
impl CachedGauss for DoubleMt {
    fn push_gauss(&mut self, val: f64) -> bool {
        if self.cached_gauss.is_none() {
            self.cached_gauss.replace(val);
            true
        } else {
            false
        }
    }
    fn pop_gauss(&mut self) -> Option<f64> {
        self.cached_gauss.take()
    }
}
impl DoubleMt {
    fn new(seed: Option<u32>) -> Self {
        let seed = seed.unwrap_or_else(get_random_seed);
        DoubleMt {
            mt: Mt::new(seed),
            cached_gauss: None,
        }
    }
}
impl RngCore for DoubleMt {
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        self.mt.fill_bytes(dst);
    }
    fn next_u32(&mut self) -> u32 {
        self.mt.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.mt.next_u64()
    }
}

fn rng_double<R: RngCore>(rng: &mut R) -> f64 {
    let a = rng.next_u32() >> 5;
    let b = rng.next_u32() >> 6;
    (a as f64 * 67108864.0 + b as f64) / 9007199254740992.0
}

fn normal_marsaglia2<R: RngCore + CachedGauss>(rng: &mut R) -> f64 {
    if let Some(gg) = rng.pop_gauss() {
        return gg;
    }
    let mut y: f64;
    let mut r2: f64;
    let mut x: f64;
    while {
        let xu = rng_double(rng);
        let yu = rng_double(rng);
        x = 2.0f64 * xu - 1.0;
        y = 2.0f64 * yu - 1.0;
        r2 = x * x + y * y;
        r2 > 1.0 || r2 == 0.0
    } {}
    let f = (-2.0 * r2.ln() / r2).sqrt();
    rng.push_gauss(x * f);
    y * f
}
fn rng_norm_vec<R: RngCore + CachedGauss>(rng: &mut R, size: usize, sig: f64) -> Vec<f64> {
    (0..size).map(|_i| normal_marsaglia2(rng) * sig).collect()
}
fn rng_norm_array<R: RngCore + CachedGauss, Sh: ShapeBuilder<Dim = D>, D: Dimension>(
    rng: &mut R,
    size: Sh,
    sig: f64,
) -> Array<f64, D> {
    let shape: Shape<D> = size.f();
    let n = shape.size();
    Array::from_shape_vec(shape, rng_norm_vec(rng, n, sig)).unwrap()
}
pub fn normal_mt_impl<Sh: ShapeBuilder<Dim = D>, D: Dimension>(
    seed: Option<u32>,
    size: Sh,
    scale: f64,
) -> Array<f64, D> {
    // let mut out: Array<f64, D> = Array::default(size);
    let mut rng = DoubleMt::new(seed);

    rng_norm_array(&mut rng, size, scale)
}
