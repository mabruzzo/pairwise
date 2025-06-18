struct Data {
    // generalize these to be the registers required by the accumulator
    // somehow...
    weight: Vec<f64>,
    total: Vec<f64>,
}

struct Mean {}

impl Mean {
    pub fn initialize(&self, data: &mut Data) {
        data.weight.fill(0.0);
        data.total.fill(0.0);
    }

    pub fn apply(&self, data: &mut Data, val: f64, weight: f64, index: usize) {
        data.weight[index] += weight;
        data.total[index] += val * weight;
    }

    pub fn postprocess(&self, data: &mut Data) {
        for i in 0..data.weight.len() {
            // TODO need to think about divide by 0
            data.total[i] / data.weight[i];
        }
    }

    // TODO consider default implementation
    pub fn merge(&self, data: &mut Data, other: &Data) {
        for i in 0..data.weight.len() {
            data.total[i] += other.total[i];
            data.weight[i] += other.total[i];
        }
    }
}

struct Accumulator {
    kernel: Mean,
    data: Data,
}

/*
 * pub fn apply_accumulator(acc, input1, input2) {
 *     acc.initialize()
 *     for all the pairs of points:
 *        acc.apply(input1, input2)
 *     acc.postprocess()
 *     return acc.result()
 * }
 */

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
