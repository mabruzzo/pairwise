use criterion::{
    AxisScale, BatchSize, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};

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
    single_tile: bool,
) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

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
    group.plot_config(plot_config);

    for i in input_array.iter().cloned() {
        let n_elements = (i as u64).pow(3);
        let shape = [i, i, i];
        let dims = [5.0, 5.0, 5.0];
        let data = TestDataWrapper::from_random(shape, dims, 2525365464_u64);

        let (n_pairs, data_other) = if single_tile {
            let n_pairs = n_elements * (n_elements - 1);
            (n_pairs, None)
        } else {
            let n_pairs = n_elements * n_elements;
            let data_other = Some(TestDataWrapper::from_random(shape, dims, 57345783_u64));
            (n_pairs, data_other)
        };

        let data_pair = (data, data_other.as_ref());

        group.throughput(Throughput::Elements(n_pairs));
        group.bench_with_input(
            BenchmarkId::new("Cartesian", i),
            &data_pair,
            |b, data_pair: &(TestDataWrapper, Option<&TestDataWrapper>)| {
                b.iter_batched_ref(
                    setup_fn,
                    |accum: &mut Accumulator| {
                        process_cartesian(
                            accum,
                            data_pair.0.cartesian_block(),
                            data_pair.1.map(|t: &TestDataWrapper| t.cartesian_block()),
                            data_pair.0.cell_width(),
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
            &data_pair,
            |b, data_pair: &(TestDataWrapper, Option<&TestDataWrapper>)| {
                b.iter_batched_ref(
                    setup_fn,
                    |accum: &mut Accumulator| {
                        process_unstructured(
                            accum,
                            data_pair.0.point_props(),
                            data_pair.1.map(|t: &TestDataWrapper| t.point_props()),
                            &RuntimeSpec,
                        )
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
    help_setup_criterion_benchmark(c, "1tile", "astro_sf1", params, true);
    help_setup_criterion_benchmark(c, "1tile", "2pcf", params, true);
    let params = &[4, 5, 6];
    help_setup_criterion_benchmark(c, "2tile", "astro_sf1", params, false);
    help_setup_criterion_benchmark(c, "2tile", "2pcf", params, false);
}

criterion_group!(benches, parameterized_benchmark);
criterion_main!(benches);
