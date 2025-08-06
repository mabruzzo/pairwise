/// Specifies the kind of pairwise operation to do in a two-point operation.
///
/// # Note
/// We're going to try to get away with making this an enum. But, it may be
/// necessary to convert this to a trait (each variant would become a
/// unit-like struct)
#[derive(Clone, Copy)]
pub enum PairOperation {
    /// For a pair of vector measurements, compute the element-wise
    /// products. This is used to compute correlation functions
    ElementwiseMultiply,
    /// For a pair of vector measurements, compute the element-wise
    /// differences. This is used to compute structure functions
    ElementwiseSub,
    // /// For a pair of vector measurements, compute the element-wise
    // /// differences and then take the dot-product of the resulting vector
    // /// and the separation vector
    // ElementwiseSubLongitudinalComp
}
