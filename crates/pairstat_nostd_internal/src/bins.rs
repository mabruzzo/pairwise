//! Implements types to represent "bin edges", used for Histogram buckets and for
//! distance binning of accumulators.  The [`BinEdges`] trait provides a common
//! interface that is implemented by [`RegularBinEdges`] and [`IrregularBinEdges`]

/// Super simple. This can be expanded as needed.
pub trait BinEdges {
    /// Calculate the bin index for a given value. Values which are equal to
    /// boundary values are considered part of the higher bin, i.e. intervals
    /// do not include the right edge.
    fn bin_index(&self, value: f64) -> Option<usize>;

    fn n_bins(&self) -> usize;

    fn leftmost_edge(&self) -> f64;
}

/// Regular bins with uniform spacing
#[derive(Clone)]
#[cfg_attr(feature = "fmt", derive(Debug))]
pub struct RegularBinEdges {
    min: f64,
    max: f64,
    bin_size: f64,
    n_bins: usize,
}
impl RegularBinEdges {
    /// Note that we initialize with num_bins rather than bin_size
    pub fn new(min: f64, max: f64, n_bins: usize) -> Result<Self, &'static str> {
        let diff = max - min;
        let bin_size = diff / (n_bins as f64);
        if n_bins == 0 {
            Err("Number of bins must be greater than zero")
        } else if max <= min {
            Err("Maximum value must be greater than minimum value")
        } else if !diff.is_finite() {
            // also catches pathological case: (max - min) > f64::MAX
            Err("Min and max values must be finite")
        } else if bin_size == 0.0 {
            // (diff / n_bins) < f64::MIN
            Err("received pathological values")
        } else {
            Ok(Self {
                min,
                max,
                bin_size,
                n_bins,
            })
        }
    }
}

// we can't derive `Eq` for RegularBinEdges because f64 doesn't implement the
// trait (since `NaN`!=`NaN`). Because we confirmed in the constructor that
// both min and max are finite, we can manually implement the trait
impl PartialEq for RegularBinEdges {
    fn eq(&self, other: &Self) -> bool {
        // we can skip over bin_size since it is derived from the other parameters
        (self.min == other.min) && (self.max == other.max) && (self.n_bins == other.n_bins)
    }
}

impl Eq for RegularBinEdges {}

impl BinEdges for RegularBinEdges {
    fn bin_index(&self, value: f64) -> Option<usize> {
        if value < self.min || value >= self.max {
            return None;
        }

        // this cast handles the truncation
        let index = ((value - self.min) / self.bin_size) as usize;

        Some(index)
    }

    fn n_bins(&self) -> usize {
        self.n_bins
    }

    fn leftmost_edge(&self) -> f64 {
        self.min
    }
}

pub fn validate_bin_edges(edges: &[f64]) -> Result<(), &'static str> {
    if edges.len() < 2 {
        Err("A minimum of two bin edges are required")
    } else if edges.iter().any(|&x| !x.is_finite()) {
        // Check that all bin_edges are finite
        // It may be worth supporting -inf and +inf as first and last bin edges
        Err("Bin edges must be finite")
    } else if edges
        .iter()
        .zip(edges.iter().skip(1))
        .any(|(&left, &right)| right <= left)
    {
        // Check if bin_edges are in strictly increasing order
        Err("Bin edges must be in strictly increasing order")
    } else {
        Ok(())
    }
}

#[derive(Clone)]
#[cfg_attr(feature = "fmt", derive(Debug))]
pub struct IrregularBinEdges<'a> {
    bin_edges: &'a [f64],
}

impl<'a> IrregularBinEdges<'_> {
    pub fn new(bin_edges: &'a [f64]) -> Result<IrregularBinEdges<'a>, &'static str> {
        validate_bin_edges(bin_edges)?;
        Ok(IrregularBinEdges { bin_edges })
    }
}

impl BinEdges for IrregularBinEdges<'_> {
    fn bin_index(&self, value: f64) -> Option<usize> {
        if value < self.bin_edges[0] || value >= self.bin_edges[self.bin_edges.len() - 1] {
            return None;
        }

        let index = self
            .bin_edges
            // There may be downsides to using total_cmp (perf?)
            .binary_search_by(|probe| probe.total_cmp(&value))
            // Ok is used for an exact match, Err for a lower bound
            .unwrap_or_else(|i| i - 1);

        Some(index)
    }

    fn n_bins(&self) -> usize {
        self.bin_edges.len() - 1
    }

    fn leftmost_edge(&self) -> f64 {
        self.bin_edges[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_bins_invalid_creation() {
        // Zero bins
        assert!(RegularBinEdges::new(0.0, 10.0, 0).is_err());

        // Max <= min
        assert!(RegularBinEdges::new(10.0, 10.0, 5).is_err());
        assert!(RegularBinEdges::new(10.0, 5.0, 5).is_err());

        // Non-finite values
        assert!(RegularBinEdges::new(f64::NAN, 10.0, 5).is_err());
        assert!(RegularBinEdges::new(0.0, f64::INFINITY, 5).is_err());
    }

    #[test]
    fn irregular_bins_invalid_creation() {
        // not enough edges
        assert!(IrregularBinEdges::new(&[0.0]).is_err());

        // unsorted bin edges
        assert!(IrregularBinEdges::new(&[2.0, 1.0]).is_err());
        assert!(IrregularBinEdges::new(&[0.0, 3.0, 2.0]).is_err());

        // Non-finite values
        assert!(IrregularBinEdges::new(&[f64::NAN, 10.0]).is_err());
        assert!(IrregularBinEdges::new(&[0.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn regular_and_irregular_bin_indexing() {
        let rbins = RegularBinEdges::new(0.0, 10.0, 5).unwrap();
        let ibins = IrregularBinEdges::new(&[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]).unwrap();

        let bins_list: [&dyn BinEdges; 2] = [&rbins, &ibins];

        for bins in &bins_list {
            assert_eq!(bins.n_bins(), 5);

            // Test valid values
            assert_eq!(bins.bin_index(0.0), Some(0));
            assert_eq!(bins.bin_index(1.9), Some(0));
            assert_eq!(bins.bin_index(2.0), Some(1));
            assert_eq!(bins.bin_index(3.9), Some(1));
            assert_eq!(bins.bin_index(4.0), Some(2));
            assert_eq!(bins.bin_index(8.0), Some(4));
            assert_eq!(bins.bin_index(9.9), Some(4));

            // Test boundary conditions
            assert_eq!(bins.bin_index(10.0), None); // max is exclusive
            assert_eq!(bins.bin_index(-0.1), None); // below min
            assert_eq!(bins.bin_index(10.1), None); // above max
        }
    }

    #[test]
    fn irregular_bins_bin_indexing() {
        let bins = IrregularBinEdges::new(&[-5.0, 0.0, 2.0, 3.0]).unwrap();

        assert_eq!(bins.n_bins(), 3);

        // Test valid values
        assert_eq!(bins.bin_index(-5.0), Some(0));
        assert_eq!(bins.bin_index(-2.5), Some(0));
        assert_eq!(bins.bin_index(-0.1), Some(0));
        assert_eq!(bins.bin_index(0.0), Some(1));
        assert_eq!(bins.bin_index(1.0), Some(1));
        assert_eq!(bins.bin_index(1.9), Some(1));
        assert_eq!(bins.bin_index(2.0), Some(2));
        assert_eq!(bins.bin_index(2.5), Some(2));
        assert_eq!(bins.bin_index(2.9), Some(2));

        // Test boundary conditions
        assert_eq!(bins.bin_index(3.0), None); // max is exclusive
        assert_eq!(bins.bin_index(-5.1), None); // below min
        assert_eq!(bins.bin_index(3.1), None); // above max
    }
}
