// I don't really know what I'm doing with error handling quite yet here.
//
// But, I think its time to start doing "something" so we can start digging
// our way out of the "hole" that we created in the `pairwise_nostd_internal`
// by returning `&'static str` everywhere...
//
// It seems like we basically have 2 options:
// 1. we can do all error definitions within the internal crate and make
//    everything public
// 2. we can define separate Error types within the publib and internal crate
//    and just have the public crate wrap the internal crate.
//
// I think we're going to start with option #2. Even if it introduces
// marginally more upfront work, it is the more flexible approach.
// Importantly, it is **MUCH** easier to migrate from approach #2 to approach
// #1.
//
// The jiff crate has a whole discussion about error types. It merits further
// review!

#[derive(Debug)]
pub struct Error {
    // I'm not so sure we want to directly expose this
    kind: ErrorKind,
    // do we want to track a cause?
    //cause: Option<Error>,
}

/// The underlying internal error type
#[non_exhaustive]
#[derive(Clone, Debug)]
enum ErrorKind {
    /// An error that occurs when a binned_statepack has the wrong shape
    BinnedStatePackShape(BinnedStatePackShapeError),
    /// An error related to specifying (or not specifying) the Bin Edges
    /// for the Buckets in a Histogram Reducer
    BucketEdge(BucketEdgeError),
    /// An error that occurs when a problematic distance bin edge is specified
    DistanceEdge(DistanceEdgeError),
    /// An error that occurs when an integer lies outside of the acceptable
    /// range of values
    IntegerRanger(IntegerRangeError),
    /// An error that occurs within `pairwise_nostd_internal`
    ///
    /// The idea is to wrap the stringly errors that, at the time of writing,
    /// are pervasive within `pairwise_nostd_internal`. Over time, these will
    /// all get migrated over to InternalError
    #[allow(dead_code)] // not used yet
    InternalLegacyAdHoc(InternalLegacyAdHocError),
    // /// An error that occurs within `pairwise_nostd_internal`
    // ///
    // /// The idea is to wrap the error type introduced within
    // /// `pairwise_nostd_internal`, whenever that actually gets introduced...
    // InternalError(InternalError)
    /// An error that occurs when distance bin edges aren't specified
    MissingDistanceEdge(MissingDistanceEdgeError),
    /// An error that occurs when an unknown reducer name is specified
    ReducerName(ReducerNameError),
}

// define constructor methods for Error
impl Error {
    /// produce an error indicating that a binned_statepack has the wrong shape
    pub(crate) fn binned_statepack_shape(
        expected_n_states: u64,
        expected_accum_size: u64,
        actual_n_states: u64,
        actual_accum_size: u64,
    ) -> Self {
        Error {
            kind: ErrorKind::BinnedStatePackShape(BinnedStatePackShapeError {
                expected_n_states,
                expected_accum_size,
                actual_n_states,
                actual_accum_size,
            }),
        }
    }

    /// produce an error indicating the presence/ommision of the bucket
    /// bin-edges for configuring the Reducer within an Accumulator
    pub(crate) fn bucket_edges(name: String, expect_edges: bool) -> Self {
        Error {
            kind: ErrorKind::BucketEdge(BucketEdgeError { name, expect_edges }),
        }
    }

    /// produce an error indicating that a problematic distance bin edge was specified
    pub(crate) fn distance_edge(what: String) -> Self {
        Error {
            kind: ErrorKind::DistanceEdge(DistanceEdgeError { what }),
        }
    }

    /// produce an error indicating that an integer lies outside the acceptable
    /// range of values
    pub(crate) fn integer_range(
        description: &'static str,
        actual: i64,
        min_val: i64,
        max_val: i64,
    ) -> Self {
        Error {
            kind: ErrorKind::IntegerRanger(IntegerRangeError {
                description,
                actual,
                min_val,
                max_val,
            }),
        }
    }

    /// wraps a legacy internal error string
    #[allow(dead_code)] // not used yet
    pub(crate) fn internal_legacy_adhoc(message: &'static str) -> Self {
        Error {
            kind: ErrorKind::InternalLegacyAdHoc(InternalLegacyAdHocError(message)),
        }
    }

    /// produce an error indicating that distance bin edges aren't specified
    pub(crate) fn missing_distance_edge() -> Self {
        Error {
            kind: ErrorKind::MissingDistanceEdge(MissingDistanceEdgeError),
        }
    }

    /// produce an error indicating that an unknown reducer name was specified
    pub(crate) fn reducer_name(actual: String, choices: Vec<String>) -> Self {
        Error {
            kind: ErrorKind::ReducerName(ReducerNameError { actual, choices }),
        }
    }
}

impl std::error::Error for Error {}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        self.kind.fmt(f)
    }
}

impl std::error::Error for ErrorKind {}

impl core::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match *self {
            ErrorKind::BinnedStatePackShape(ref err) => err.fmt(f),
            ErrorKind::BucketEdge(ref err) => err.fmt(f),
            ErrorKind::DistanceEdge(ref err) => err.fmt(f),
            ErrorKind::IntegerRanger(ref err) => err.fmt(f),
            ErrorKind::InternalLegacyAdHoc(ref msg) => msg.fmt(f),
            ErrorKind::MissingDistanceEdge(ref err) => err.fmt(f),
            ErrorKind::ReducerName(ref err) => err.fmt(f),
        }
    }
}

/// An error that occurs when a binned_statepack has the wrong shape
///
/// # Note
/// I suspect that in the long-term, we may design our API so that this error
/// isn't directly accessible
#[derive(Clone, Debug)]
struct BinnedStatePackShapeError {
    expected_n_states: u64,
    expected_accum_size: u64,
    actual_n_states: u64,
    actual_accum_size: u64,
}

impl std::error::Error for BinnedStatePackShapeError {}

impl core::fmt::Display for BinnedStatePackShapeError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "Binned statepack has {} states & each state holds {} values. \
             It should have {} states, with {} entries per state",
            self.actual_n_states,
            self.actual_accum_size,
            self.expected_n_states,
            self.expected_accum_size
        )
    }
}

/// An error related to specifying (or not specifying) the Bin Edges
/// for the Buckets in a Histogram Reducer
#[derive(Clone, Debug)]
struct BucketEdgeError {
    name: String,
    expect_edges: bool,
}

impl std::error::Error for BucketEdgeError {}

impl core::fmt::Display for BucketEdgeError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let BucketEdgeError { name, expect_edges } = self;
        if *expect_edges {
            write!(f, "The \"{name}\" reducer didn't recieve bucket edges")
        } else {
            write!(f, "The \"{name}\" reducer shouldn't recieve bucket edges")
        }
    }
}

/// An error that occurs when an integer lies outside of the acceptable
/// range of values
#[derive(Clone, Debug)]
struct IntegerRangeError {
    description: &'static str,
    actual: i64,
    min_val: i64,
    max_val: i64,
}

impl std::error::Error for IntegerRangeError {}

impl core::fmt::Display for IntegerRangeError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "{} has a value of {}. The value should be no less than {} and \
             not exceed {}",
            self.description, self.actual, self.min_val, self.max_val
        )
    }
}

/// A temporary type (that will eventually be eliminated) that wraps the
/// string errors from `pairwise_nostd_internal`. Over time, we will get rid
/// of these
#[derive(Clone)]
struct InternalLegacyAdHocError(&'static str);

impl std::error::Error for InternalLegacyAdHocError {}

impl core::fmt::Display for InternalLegacyAdHocError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.0, f)
    }
}

impl core::fmt::Debug for InternalLegacyAdHocError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Debug::fmt(&self.0, f)
    }
}

/// An error that occurs when distance bin edges aren't specified
#[derive(Clone, Debug)]
struct MissingDistanceEdgeError;

impl std::error::Error for MissingDistanceEdgeError {}

impl core::fmt::Display for MissingDistanceEdgeError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "distance bin edges were not specified")
    }
}

/// An error that occurs when a problematic distance bin edge is specified
#[derive(Clone, Debug)]
struct DistanceEdgeError {
    what: String,
}

impl std::error::Error for DistanceEdgeError {}

impl core::fmt::Display for DistanceEdgeError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let what = self.what.as_str();
        write!(f, "problem with squared distance bin: {what}")
    }
}

/// An error occurs when an unknown reducer name is specified
#[derive(Clone, Debug)]
struct ReducerNameError {
    actual: String,
    choices: Vec<String>,
}

impl std::error::Error for ReducerNameError {}

impl core::fmt::Display for ReducerNameError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "{} is not a reducer name. Choices include: {:?}",
            self.actual, self.choices
        )
    }
}
