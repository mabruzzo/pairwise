# pairstat-rs
[![codecov](https://codecov.io/gh/mabruzzo/pairwise/graph/badge.svg?token=56Z7H1GNK8)](https://codecov.io/gh/mabruzzo/pairwise)

An experimental rust package that provides low-level functionality for computing two-point statistics (e.g. correlation functions and structure functions).

> [!NOTE]
> We are in the process of renaming the crate from pairwise to pairstat in order to improve searchability (pairwise testing is a common term)

Ideally, the goal is to replace the C++ code within [pyvsf](https://github.com/mabruzzo/pyvsf) with bindings to this crate (this will coincide an overhaul to the API).


> [!NOTE]
> The remainder of this document describes some assorted ideas. They probably need to be distributed within actual documentation before we make a formal release.

# Generating Documentation Locally

After cloning the repository, navigate to the root of the repository and invoke

```shell
$ cargo doc
```

You can find the generated documentation within **target/doc**. You can manually open the docs for the ``pairwise`` crate, **OR**, you can invoke ``cargo doc --open``.


# Design Considerations

This package's design is influenced by the different contexts where you might compute such two-point statistics.

In general, we identify 2 contexts:

1. *The simple context,* where all the data involved in the calculation is easy to put into the memory of a single CPU.
  - This context is directly supported by this package
  - In this scenario, an application/script calls a single function provided by this package that does all the heavy lifting. Behind the scenes, this package implements parallelism strategies

2. *The more general context,* where it isn't so easy to put the data into the memory of a single CPU.
  - This package doesn't provide functions to directly handle this scenario
  - Instead, this package designs its functionality in such a way that it can be used as building-blocks for an external application/library to implement a solution.
  - This scenario is quite common in astrophysical simulations.

## Elaborating on the General Context

> [!IMPORTANT]
> Is the way that we use auto-correlation and cross-correlation confusing? Should we be calling the operations intra-tile correlations and inter-tile correlations? (Or maybe auto-correlation and inter-tile correlation?)

Suppose we have a hydrodynamical simulation, where the domain is broken up into equal sized chunks, and we want to compute a structure function or an auto-correlation for a subset of gas in the domain.

For the sake of clarity, let's provide some concrete details about this scenario. Suppose that:
- we are just interested in the auto-correlation **NOTE:** it doesn't actually matter whether we want to know about correlation or structure functions (it is slightly more convenient to describe correlation since auto-correlation and cross-correlation are familiar topics).
- the simulation is 2D. **NOTE:** the more common scenario is probably 3D. We consider 2D for pedagogical purposes (i.e. the "bookkeeping" is simpler), but all the concepts generalize to 3D as well.
- the domain is split into 12 rectangular tiles, where each tiles holds data representing a spatial region with width `tile_width` and a height `tile_height`. The tiles are organized such that the domain's total width is `3*tile_width` and the height is `4*tile_height`.
- we are only interested in the structure function for gas in a particular temperature range (say, `5e3K <= t < 2e4`).


We provide an illustration of the domain down below. In this illustrate, we assign each tile a name of the form `t[a,b]`.
```
┌────────┬────────┬────────┐
│ t[0,0] │ t[0,1] │ t[0,2] │
├────────┼────────┼────────┤
│ t[1,0] │ t[1,1] │ t[1,2] │
├────────┼────────┼────────┤
│ t[2,0] │ t[2,1] │ t[2,2] │
├────────┼────────┼────────┤
│ t[3,0] │ t[3,1] │ t[3,2] │
└────────┴────────┴────────┘
```

It's important to emphasize that the way gas from the selected temperature bin is organized within a tile (e.g. at arbitrary locations or organized in a regular Cartesian grid) **is irrelevant to this discussion.**

The global auto-correlation of this grid can be recast as the combined result of a series of auto-correlations within tiles and cross-correlations between tiles. We enumerate these operations below, assuming that the max correlation-length does not exceed `min(tile_width, tile_height)` and that the domain is aperiodic:

* `t[0,0]-auto` `t[0,0]-[0,1]` `t[0,0]-[1,1]` `t[0,0]-[1,0]`
* `t[0,1]-auto` `t[0,1]-[0,2]` `t[0,1]-[1,2]` `t[0,1]-[1,1]` `t[1,1]-[1,0]`
* `t[0,2]-auto`                               `t[0,2]-[1,2]` `t[0,2]-[1,1]`
* `t[1,0]-auto` `t[1,0]-[1,1]` `t[1,0]-[2,1]` `t[1,0]-[2,0]`
* `t[1,1]-auto` `t[1,1]-[1,2]` `t[1,1]-[2,2]` `t[1,1]-[2,1]` `t[1,1]-[2,0]`
* `t[1,2]-auto`                               `t[1,2]-[2,2]` `t[1,2]-[2,1]`
* `t[2,0]-auto` `t[2,0]-[2,1]` `t[2,0]-[3,1]` `t[2,0]-[3,0]`
* `t[2,1]-auto` `t[2,1]-[2,2]` `t[2,1]-[3,2]` `t[2,1]-[3,1]` `t[2,1]-[3,0]`
* `t[2,2]-auto`                               `t[2,2]-[3,2]` `t[2,2]-[3,1]`
* `t[3,0]-auto` `t[3,0]-[3,1]`
* `t[3,1]-auto` `t[3,1]-[3,2]`
* `t[3,2]-auto`

In the above scenario, up to a given tile is involved in 1 auto-correlation and up to 8 cross-tile correlations. If we had a periodic simulation domain, then all tiles would be involved in 8 cross-tile correlations. This pattern generalizes to 3D, with a few minor modifications.[^generalize3D]

The advantage of this sort of decomposition is that you don't need to load all the data into memory at once and that it's easy to distribute work between nodes when using a HPC cluster. If you arrange your calculation such that you have multiple cores per node, you can parallelize each individual auto-correlation and cross-tile correlation.

Distributing this work across nodes of an HPC cluster may seem a little contrived because this is a somewhat simplified example of a real world scenario.
- In the real world, you could plausibly have a 3D simulation distributed across 16^3 tiles.
- Moreover, you might be interested in computing global auto-correlations for different subsets of gas in the simulation (e.g. gas that falls into different temperature bins).
  - In general, these are totally independent calculations that could be performed entirely separately.
  - But, given the performance challenges with parallel file systems, there may be **significant** performance improvements if you interleave the calculations to minimize the number of times files need to be accessed.


## How the scenario pertains to our package?

At the time of writing, the package **does not** directly provide a function for performing this sort of distributed calculation.
Instead the package implements functionality in a manner that it can be used as building blocks for an external library/application that does support this.

### How it pertains to the package's design?

The most obvious way this influences design is that we implement functionality in terms of accumulators.
- If we didn't care about this scenarios, all of this package's functions would **directly** return the result of a structure function calculation or a correlation calculation.
- Instead, the package's primary public functions are designed to work with accumulators.
  - The general flow is that a user initializes an accumulator for their chosen statistic, calls a function that updates the accumulator's internal state with the contributions from a given set of point-pairs, and calls a method to convert the internal accumulator state to the desired statistic.
  - The package also makes it easy to combine contributions from multiple accumulators
  - We also plan to make it easy to serialize the accumulator's state
  - All of this together should make it very easy to combine contributions from various auto and inter-tile calculations

We also have plans to offer some functionality to compute the necessary inter-tile correlations that a caller requires for an arbitrary tiling of the simulation domain and an arbitrary set of separation bins. We plan to provide this functionality because

- this bookkeeping is tricky and easy to mess up
- internally, we already need to implement something similar (so hopefully we can reuse functionality)
- we can probably use this to help with periodic boundaries

### Why the package doesn't directly implement distributed calculations?

This decision is largely a matter of scope. Writing machinery to support this scenario, that is general purpose enough to be broadly useful in different contexts, would constitute a significant amount of additional development time. At the time of writing, it doesn't seem worthwhile given the number of contributors. Importantly, it also isn't necessary for the package's core functionality to have awareness about this sort of distributed parallelism; this sort of distributed parallelism can be implemented separately with minimal impact on performance.[^distributed_parallelism]

With that said, if you think such a function would be useful and are interested in contributing to or spearheading its development, please reach out to us!

If we added this functionality, here are some important considerations:

- to be broadly useful for the computational astrophysics community, such functionality probably needs to make it possible to provide python callbacks to read in data.
- If we choose to support that, it's worth evaluating if makes more sense to implement function's logic directly in Python (and dispatch to this package for the actual correlation calculations)?
  - Compared to this functionality already defined within this package, the requirements and purpose of this functionality are much more open-ended. Until it reaches a certain level of feature-completion, such functionality may require minor contributions from users to actually be useful. It may therefore be important that python is a more accessible language
  - Using python could potentially make it easier for us be more agnostic about how exactly internode communication is managed (e.g. it *might* be easier to integrate with existing analysis frameworks). In practice, I have doubts about whether this is true and how relevant this actually is.
  - At the same time, Rust's powerful type system could be useful in this context. Plus it could potentially be convenient to have lower-level access to machinery with the package (in particular, we could potentially reuse internal functionality that we don't want to commit to supporting as a part of a public API)
  - Using rust would let us expose the functions to languages other than python much more easily (e.g. if you wanted to embed the function to perform online analysis inside of a simulation). In practice, I have doubts about how relevant this actually is.
- if this function were implemented in rust, I think it's very important that we do it in such a way that the function (and any dependency-packages it requires) are an optional feature. To elaborate:
  - we would probably want to use MPI for distributed communications (due to its ubiquity). However, such a choice would prevent us from providing this function in a pre-compiled binary because different MPI implementations are **NOT** ABI-compatible.
  - in particular, we want to make sure that any python package that depends on the pairwise package can be distributed as a precompiled binary.


[^generalize3D]: The number of cross-tile correlations increase in 3D. If the max correlation-length does not exceed `min(tile_width, tile_height, tile_length)`, then a given tile may be involved with up to 26 cross-tile correlations.

[^distributed_parallelism]: Honestly, even if we did support this sort of distributed parallelism as part of the pairwise crate, it would probably be handled somewhat separately from the core functions that do the heavy-lifting of computing two-point statistics (in order to avoid introducing unnecessary additional complexity -- thread-teams are already complex enough).
