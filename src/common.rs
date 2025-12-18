use numpy::ndarray::prelude::*;
use std::borrow::Cow;

pub type RsEMDOut = (Array2<f64>, Array1<f64>);
pub(crate) fn get_cow_slice<'a, T>(x: &'a ArrayView1<T>) -> Cow<'a, [T]>
where
    T: Clone,
    [T]: ToOwned<Owned = Vec<T>>,
{
    match x.as_slice() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(x.to_vec()),
    }
}
