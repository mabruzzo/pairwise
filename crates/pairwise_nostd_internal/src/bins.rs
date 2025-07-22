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
}

/// Regular bins with uniform spacing
#[derive(Clone)]
pub struct RegularBins {
    min: f64,
    max: f64,
    bin_size: f64,
}
impl RegularBins {
    /// Note that we initialize with num_bins rather than bin_size
    pub fn new(min: f64, max: f64, num_bins: usize) -> Result<Self, &'static str> {
        if num_bins == 0 {
            Err("Number of bins must be greater than zero")
        } else if max <= min {
            Err("Maximum value must be greater than minimum value")
        } else if !min.is_finite() || !max.is_finite() {
            Err("Min and max values must be finite")
        } else {
            Ok(Self {
                min,
                max,
                bin_size: (max - min) / num_bins as f64,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regular_bins_invalid_creation() {
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
    fn test_regular_bins_bin_index() {
        let bins = RegularBins::new(0.0, 10.0, 5).unwrap();

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
