use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use pairstat::{
    Accumulator, AccumulatorBuilder, RuntimeSpec, process_cartesian, process_unstructured,
};
use pairstat_test::TestDataWrapper;

// this was thrown together in a sloppy manner. we can definitely do better!

fn help_setup_criterion_benchmark(
    c: &mut Criterion,
    prefix: &str,
    kind: &str,
    input_array: &[usize],
) {
    let setup_fn = || -> Accumulator {
        AccumulatorBuilder::new()
            .calc_kind(kind)
            .dist_bin_edges(&[0.0, 2., 4., 6.])
            .build()
            .unwrap()
    };

    let mut name = prefix.to_owned();
    name.push('-');
    name.push_str(kind);

    let mut group = c.benchmark_group(name);
    for i in input_array.iter().cloned() {
        let n_elements = (i as u64).pow(3);
        let test_data = TestDataWrapper::from_random([i, i, i], [5.0, 5., 5.0], 2525365464_u64);

        group.throughput(Throughput::Elements(n_elements));
        group.bench_with_input(
            BenchmarkId::new("Cartesian", i),
            &test_data,
            |b, test_data: &TestDataWrapper| {
                b.iter_batched_ref(
                    setup_fn,
                    |accum: &mut Accumulator| {
                        process_cartesian(
                            accum,
                            test_data.cartesian_block(),
                            None,
                            test_data.cell_width(),
                            &RuntimeSpec,
                        )
                    },
                    BatchSize::LargeInput, // we may be able to use BatchSize::SmallInput
                )
            },
        );

        // add UnstructuredPoints
        group.bench_with_input(
            BenchmarkId::new("Unstructured", i),
            &test_data,
            |b, test_data: &TestDataWrapper| {
                b.iter_batched_ref(
                    setup_fn,
                    |accum: &mut Accumulator| {
                        process_unstructured(accum, test_data.point_props(), None, &RuntimeSpec)
                    },
                    BatchSize::LargeInput, // we may be able to use BatchSize::SmallInput
                )
            },
        );
    }
    group.finish();
}

fn parameterized_benchmark(c: &mut Criterion) {
    let params = &[4, 5, 6, 7, 8];
    help_setup_criterion_benchmark(c, "parameterized", "astro_sf1", params);
    help_setup_criterion_benchmark(c, "parameterized", "2pcf", params);
}

criterion_group!(benches, parameterized_benchmark);
criterion_main!(benches);
