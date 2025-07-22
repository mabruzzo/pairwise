//! Implements types to represent "bins", used for Histogram buckets and for
//!  distance binning of accumulators.  The Bins trait provides a common
//! interface which is implemented by RegularBins and IrrigularBins. These
//! types do NOT "hold" data, they only define the bin edges and determine the
//! bin index for a given value.

/// Super simple. This can be expanded as needed.
pub trait Bins {
    /// Calculate the bin index for a given value. Values which are equal to
    /// boundary values are considered part of the higher bin.
    fn bin_index(&self, value: f64) -> Option<usize>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Regular bins with uniform spacing
#[derive(Clone)]
pub struct RegularBins {
    min: f64,
    max: f64,
    bin_size: f64,
    n_bins: usize,
}
impl RegularBins {
    /// Note that we initialize with num_bins rather than bin_size
    pub fn new(min: f64, max: f64, n_bins: usize) -> Result<Self, &'static str> {
        if n_bins == 0 {
            Err("Number of bins must be greater than zero")
        } else if max <= min {
            Err("Maximum value must be greater than minimum value")
        } else if !min.is_finite() || !max.is_finite() {
            Err("Min and max values must be finite")
        } else {
            Ok(Self {
                min,
                max,
                bin_size: (max - min) / n_bins as f64,
                n_bins,
            })
        }
    }
}

impl Bins for RegularBins {
    fn bin_index(&self, value: f64) -> Option<usize> {
        if value < self.min || value >= self.max {
            return None;
        }

        let index = ((value - self.min) / self.bin_size).floor() as usize;
        Some(index)
    }

    fn len(&self) -> usize {
        self.n_bins
    }
}

pub struct IrregularBins<'a> {
    bin_edges: &'a [f64],
}

impl<'a> IrregularBins<'_> {
    pub fn new(bin_edges: &'a [f64]) -> Result<IrregularBins<'a>, &'static str> {
        if bin_edges.len() < 2 {
            return Err("A minimum of two bin edges are required");
        }

        // Check that all bin_edges are finite
        // It may be worth supporting -inf and +inf as first and last bin edges

        if bin_edges.iter().any(|&x| !x.is_finite()) {
            return Err("Bin edges must be finite");
        }

        // Check if bin_edges are in strictly increasing order
        for i in 1..bin_edges.len() {
            if bin_edges[i] <= bin_edges[i - 1] {
                return Err("Bin edges must be in strictly increasing order");
            }
        }

        Ok(IrregularBins { bin_edges })
    }
}

impl Bins for IrregularBins<'_> {
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

    fn len(&self) -> usize {
        self.bin_edges.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_bins_invalid_creation() {
        // Zero bins
        assert!(RegularBins::new(0.0, 10.0, 0).is_err());

        // Max <= min
        assert!(RegularBins::new(10.0, 10.0, 5).is_err());
        assert!(RegularBins::new(10.0, 5.0, 5).is_err());

        // Non-finite values
        assert!(RegularBins::new(f64::NAN, 10.0, 5).is_err());
        assert!(RegularBins::new(0.0, f64::INFINITY, 5).is_err());
    }

    #[test]
    fn irregular_bins_invalid_creation() {
        // not enough edges
        assert!(IrregularBins::new(&[0.0]).is_err());

        // unsorted bin edges
        assert!(IrregularBins::new(&[2.0, 1.0]).is_err());

        // Non-finite values
        assert!(IrregularBins::new(&[f64::NAN, 10.0]).is_err());
        assert!(IrregularBins::new(&[0.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn regular_and_irregular_bin_indexing() {
        let rbins = RegularBins::new(0.0, 10.0, 5).unwrap();
        let ibins = IrregularBins::new(&[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]).unwrap();

        let bins_list: [&dyn Bins; 2] = [&rbins, &ibins];

        for bins in &bins_list {
            assert_eq!(bins.len(), 5);

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
        let bins = IrregularBins::new(&[-5.0, 0.0, 2.0, 3.0]).unwrap();

        assert_eq!(bins.len(), 3);

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
