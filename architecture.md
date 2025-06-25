# how to use (concept)

## Simple Case

(All of these names should be reconsidered)

- initialize accumulator
  - inputs: hist bin edges (maybe), number of separation bins
- set up your data view (PointsProps or CartesianBlock)
  - inputs: data, weights
- pass your accumulator and data view to the the apply_accumulator function
  - input: accumulator, data view, the separation bins
  - maybe provide apply_accumulator with some CPU-level parallelism directives
- extract the results from the accumulator

## Advanced Case

The above, but distributed over many cores (nodes?). This requires reusing and merging of accumulators.

# Types

- DataView Enum (CartesianBlock and PointsProps are held by variants or are the variants themselves?)
- AccumulatorBuilder? (adam dislikes) or some function to generically construct any specific accumulator
- Accumulator (generic with respect to the underlying type )

   enum Accumulator {
     MeanAccumulator(Mean),
     HistogramAccumulator(Histogram),
   }.

   Small chance we can work around this using `TypeId`?
- several specific accumulators
