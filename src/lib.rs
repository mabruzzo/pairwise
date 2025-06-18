use std::{str::FromStr, usize};

struct Data {
    // generalize these to be the registers required by the accumulator
    // somehow...
    weight: Vec<f64>,
    total: Vec<f64>,
}

impl Data {
    /// Create a data instance with length `n` registers
    pub fn new(n: usize) -> Result<Data, String> {
        if n == 0 {
            Err(String::from("n can't be zero"))
        } else {
            Ok(Data {
                weight: vec![0.0; n],
                total: vec![0.0; n],
            })
        }
    }
}

struct Mean {}

impl Mean {
    pub fn initialize(&self, data: &mut Data) {
        data.weight.fill(0.0);
        data.total.fill(0.0);
    }

    pub fn consume(&self, data: &mut Data, val: f64, weight: f64, index: usize) {
        data.weight[index] += weight;
        data.total[index] += val * weight;
    }

    pub fn get_value(&self, data: &Data) -> (Vec<f64>, Vec<f64>) {
        let mut mean = vec![0.0; data.weight.len()];
        for i in 0..data.weight.len() {
            // TODO need to think about divide by 0
            mean[i] = data.total[i] / data.weight[i];
        }
        (mean, data.weight.clone())
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

impl Accumulator {
    pub fn new(n: usize) -> Result<Accumulator, String> {
        Ok(Accumulator {
            kernel: Mean {},
            data: Data::new(n)?,
        })
    }

    /// Apply the accumulator to a pair of values
    pub fn consume(&mut self, val: f64, weight: f64, index: usize) {
        self.kernel.consume(&mut self.data, val, weight, index);
    }

    pub fn get_value(&self) -> (Vec<f64>, Vec<f64>) {
        self.kernel.get_value(&self.data)
    }
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consume_once() {
        let mut accum = Accumulator::new(1).unwrap();
        accum.consume(4.0, 1.0, 0_usize);
        let (mean_vec, weight_vec) = accum.get_value();
        assert_eq!(mean_vec[0], 4.0);
        assert_eq!(weight_vec[0], 1.0);
    }

    #[test]
    fn consume_twice() {
        let mut accum = Accumulator::new(1).unwrap();
        accum.consume(4.0, 1.0, 0);
        accum.consume(8.0, 1.0, 0);
        let (mean_vec, weight_vec) = accum.get_value();
        assert_eq!(mean_vec[0], 6.0);
        assert_eq!(weight_vec[0], 2.0);
    }
}
