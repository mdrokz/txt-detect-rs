use std::borrow::Borrow;

use tch::{
    nn::{conv2d, linear, seq, ConvConfig, Module, Sequential},
    Tensor,
};

use crate::utils::{batch_norm2d, AdaptiveAvgPool2d, Dropout, MaxPool2d, ReLU};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Config {
    M,
    V1 = 64,
    V2 = 128,
    V3 = 256,
    V4 = 512,
}

use crate::model::Config::{M, V1, V2, V3, V4};

const CFG: [Config; 18] = [
    V1, V1, M, V2, V2, M, V3, V3, V3, M, V4, V4, V4, M, V4, V4, V4, M,
];

fn make_layers<'p, P>(cfg: [Config; 18], batch_norm: bool, p: P) -> Sequential
where
    P: Borrow<tch::nn::Path<'p>>,
{
    // let mut layers: Vec<Box<dyn Module>> = vec![];
    let mut layers = seq();

    let mut in_channels = 3;

    let kernel_size = 3;

    let conv_config = ConvConfig {
        padding: (kernel_size - 1) / 2,
        ..Default::default()
    };

    for v in cfg {
        if v == M {
            layers = layers.add(MaxPool2d(&[2, 2], &[2, 2]));
        } else {
            let conv2d = conv2d(p.borrow(), in_channels, v as i64, kernel_size, conv_config);

            if batch_norm {
                layers = layers.add(conv2d);
                layers = layers.add(batch_norm2d(p.borrow(), v as i64, Default::default()));
                layers = layers.add(ReLU {});
            } else {
                layers = layers.add(conv2d);
                layers = layers.add(ReLU {});
            }
        }
        in_channels = v as i64;
    }

    layers
}

#[derive(Debug)]
struct VGG {
    features: Sequential,
    avgpool: AdaptiveAvgPool2d,
    classifier: Sequential,
}

impl VGG {
    pub fn new<'p, P>(p: P, features: Sequential) -> VGG
    where
        P: Borrow<tch::nn::Path<'p>>,
    {
        let mut classifer = seq();

        let avgpool = AdaptiveAvgPool2d([7, 7]);

        let linear_config = Default::default();

        classifer = classifer.add(linear(p.borrow(), 512 * 7 * 7, 4096, linear_config));

        classifer = classifer.add(ReLU {});

        classifer = classifer.add(Dropout {});

        classifer = classifer.add(linear(p.borrow(), 4096, 4096, linear_config));

        classifer = classifer.add(ReLU {});

        classifer = classifer.add(Dropout {});

        classifer = classifer.add(linear(p.borrow(), 4096, 1000, linear_config));

        VGG {
            features: features,
            avgpool: avgpool,
            classifier: classifer,
        }
    }
}

impl Module for VGG {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut x = self.features.forward(xs);

        x = self.avgpool.forward(&x);
        x = x.view_(&[x.size1().unwrap(), -1]);
        x = self.classifier.forward(&x);

        x
    }
}
