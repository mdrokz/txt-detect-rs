use std::{borrow::Borrow, f64::consts::PI, any::{Any, TypeId}};

use tch::{
    nn::{
        conv2d,
        init::{FanInOut, NonLinearity, NormalOrUniform},
        linear, seq, BatchNormConfig, Conv2D, ConvConfig, LinearConfig, Module, Sequential,
    },
    Tensor,
};

use crate::utils::{batch_norm2d, AdaptiveAvgPool2d, BatchNorm, Dropout, MaxPool2d, ReLU, Sigmoid};

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
        // padding: (kernel_size - 1) / 2,
        padding: 1,
        ws_init: tch::nn::Init::Kaiming {
            dist: NormalOrUniform::Normal,
            fan: FanInOut::FanOut,
            non_linearity: NonLinearity::ReLU,
        },
        bs_init: tch::nn::Init::Const(0.0),
        ..Default::default()
    };

    let batch_config = BatchNormConfig {
        ws_init: tch::nn::Init::Const(1.0),
        bs_init: tch::nn::Init::Const(0.0),
        ..Default::default()
    };

    for v in cfg {
        if v == M {
            layers = layers.add(MaxPool2d(&[2, 2], &[2, 2]));
        } else {
            let conv2d = conv2d(
                p.borrow() / "conv",
                in_channels,
                v as i64,
                kernel_size,
                conv_config,
            );

            if batch_norm {
                layers = layers.add(conv2d);
                layers = layers.add(batch_norm2d(
                    p.borrow() / "bn",
                    v as i64,
                    batch_config,
                ));
                layers = layers.add(ReLU {});
            } else {
                layers = layers.add(conv2d);
                layers = layers.add(ReLU {});
            }
        }
        in_channels = v as i64;
    }

    println!("layers {:?}", layers.len());

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

        let linear_config = LinearConfig {
            ws_init: tch::nn::Init::Randn {
                mean: 0.0,
                stdev: 0.001,
            },
            bs_init: Some(tch::nn::Init::Const(0.0)),
            ..Default::default()
        };

        classifer = classifer.add(linear(
            p.borrow() / "linear",
            512 * 7 * 7,
            4096,
            linear_config,
        ));

        classifer = classifer.add(ReLU {});

        classifer = classifer.add(Dropout {});

        classifer = classifer.add(linear(p.borrow() / "linear", 4096, 4096, linear_config));

        classifer = classifer.add(ReLU {});

        classifer = classifer.add(Dropout {});

        classifer = classifer.add(linear(p.borrow() / "linear", 4096, 1000, linear_config));

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
        x = x.view_(&[x.size1().expect("failed to get size of tensor"), -1]);
        println!("VGG {:?}",x);
        x = self.classifier.forward(&x);

        x
    }
}

#[derive(Debug)]
struct Merge {
    conv1: Conv2D,
    bn1: BatchNorm,
    relu1: ReLU,
    conv2: Conv2D,
    bn2: BatchNorm,
    relu2: ReLU,
    conv3: Conv2D,
    bn3: BatchNorm,
    relu3: ReLU,
    conv4: Conv2D,
    bn4: BatchNorm,
    relu4: ReLU,
    conv5: Conv2D,
    bn5: BatchNorm,
    relu5: ReLU,
    conv6: Conv2D,
    bn6: BatchNorm,
    relu6: ReLU,
    conv7: Conv2D,
    bn7: BatchNorm,
    relu7: ReLU,
}

impl Merge {
    pub fn new<'p, P>(p: P) -> Merge
    where
        P: Borrow<tch::nn::Path<'p>>,
    {
        let conv_pad_config = ConvConfig {
            padding: 1,
            ws_init: tch::nn::Init::Kaiming {
                dist: NormalOrUniform::Normal,
                fan: FanInOut::FanOut,
                non_linearity: NonLinearity::ReLU,
            },
            bs_init: tch::nn::Init::Const(0.0),
            ..Default::default()
        };

        let conv_config = ConvConfig {
            bs_init: tch::nn::Init::Const(0.0),
            ws_init: tch::nn::Init::Kaiming {
                dist: NormalOrUniform::Normal,
                fan: FanInOut::FanOut,
                non_linearity: NonLinearity::ReLU,
            },
            ..Default::default()
        };

        let batch_config = BatchNormConfig {
            ws_init: tch::nn::Init::Const(1.0),
            bs_init: tch::nn::Init::Const(0.0),
            ..Default::default()
        };

        let conv1 = conv2d(p.borrow() / "conv1", 1024, 128, 1, conv_config);
        let bn1 = batch_norm2d(p.borrow() / "bn1", 128, batch_config);
        let relu1 = ReLU {};

        let conv2 = conv2d(p.borrow() / "conv2", 128, 128, 3, conv_pad_config);
        let bn2 = batch_norm2d(p.borrow() / "bn2", 128, batch_config);
        let relu2 = ReLU {};

        let conv3 = conv2d(p.borrow() / "conv3", 384, 64, 1, conv_config);
        let bn3 = batch_norm2d(p.borrow() / "bn3", 64, batch_config);
        let relu3 = ReLU {};

        let conv4 = conv2d(p.borrow() / "conv4", 64, 64, 3, conv_pad_config);
        let bn4 = batch_norm2d(p.borrow() / "bn4", 64, batch_config);
        let relu4 = ReLU {};

        let conv5 = conv2d(p.borrow() / "conv5", 192, 32, 1, conv_config);
        let bn5 = batch_norm2d(p.borrow() / "bn5", 32, batch_config);
        let relu5 = ReLU {};

        let conv6 = conv2d(p.borrow() / "conv6", 32, 32, 3, conv_pad_config);
        let bn6 = batch_norm2d(p.borrow() / "bn6", 32, batch_config);
        let relu6 = ReLU {};

        let conv7 = conv2d(p.borrow() / "conv7", 32, 32, 3, conv_pad_config);
        let bn7 = batch_norm2d(p.borrow() / "bn7", 32, batch_config);
        let relu7 = ReLU {};

        Merge {
            conv1,
            bn1,
            relu1,
            conv2,
            bn2,
            relu2,
            conv3,
            bn3,
            relu3,
            conv4,
            bn4,
            relu4,
            conv5,
            bn5,
            relu5,
            conv6,
            bn6,
            relu6,
            conv7,
            bn7,
            relu7,
        }
    }
}

impl Module for Merge {
    fn forward(&self, xs: &Tensor) -> Tensor {
        println!("starting merge module");
        let mut y = Tensor::upsample_bilinear2d(&xs.get(3), &[], true, Some(2.0), Some(2.0));

        y = Tensor::cat(&[y, xs.get(2)], 1);


        y = self
            .relu1
            .forward(&self.bn1.forward(&self.conv1.forward(&y)));
        y = self
            .relu2
            .forward(&self.bn2.forward(&self.conv2.forward(&y)));

        y = Tensor::upsample_bilinear2d(&y, &[], true, Some(2.0), Some(2.0));

        y = Tensor::cat(&[y, xs.get(1)], 1);

        y = self
            .relu3
            .forward(&self.bn3.forward(&self.conv3.forward(&y)));
        y = self
            .relu4
            .forward(&self.bn4.forward(&self.conv4.forward(&y)));

        y = Tensor::upsample_bilinear2d(&y, &[], true, Some(2.0), Some(2.0));

        y = Tensor::cat(&[y, xs.get(0)], 1);

        y = self
            .relu5
            .forward(&self.bn5.forward(&self.conv5.forward(&y)));
        y = self
            .relu6
            .forward(&self.bn6.forward(&self.conv6.forward(&y)));
        y = self
            .relu7
            .forward(&self.bn7.forward(&self.conv7.forward(&y)));

        y
    }
}

#[derive(Debug)]
struct Output {
    scope: i64,
    conv1: Conv2D,
    sigmoid1: Sigmoid,
    conv2: Conv2D,
    sigmoid2: Sigmoid,
    conv3: Conv2D,
    sigmoid3: Sigmoid,
}

impl Output {
    pub fn new<'p, P>(p: P, scope: i64) -> Output
    where
        P: Borrow<tch::nn::Path<'p>>,
    {
        let conv_config = ConvConfig {
            ws_init: tch::nn::Init::Kaiming {
                dist: NormalOrUniform::Normal,
                fan: FanInOut::FanOut,
                non_linearity: NonLinearity::ReLU,
            },
            bs_init: tch::nn::Init::Const(0.0),
            ..Default::default()
        };

        let conv1 = conv2d(p.borrow() / "conv1", 32, 1, 1, conv_config);
        let sigmoid1 = Sigmoid {};

        let conv2 = conv2d(p.borrow() / "conv2", 32, 1, 1, conv_config);
        let sigmoid2 = Sigmoid {};

        let conv3 = conv2d(p.borrow() / "conv3", 32, 1, 1, conv_config);
        let sigmoid3 = Sigmoid {};

        Output {
            scope,
            conv1,
            sigmoid1,
            conv2,
            sigmoid2,
            conv3,
            sigmoid3,
        }
    }
}

impl Module for Output {
    fn forward(&self, xs: &Tensor) -> Tensor {
        println!("starting output module");
        let score = self.sigmoid1.forward(&self.conv1.forward(&xs));
        let loc = self.sigmoid2.forward(&self.conv2.forward(&xs)) * self.scope;
        let angle = self.sigmoid3.forward(&(&self.conv3.forward(xs) - 0.5)) * PI;

        let geo = Tensor::cat(&[loc, angle], 1);

        Tensor::cat(&[score, geo], 1)
    }
}

#[derive(Debug)]
struct Extractor {
    features: Sequential,
}

impl Extractor {
    pub fn new<'p, P>(p: P) -> Extractor
    where
        P: Borrow<tch::nn::Path<'p>>,
    {
        let vgg16_bn = VGG::new(p.borrow(), make_layers(CFG, true, p.borrow()));

        Extractor {
            features: vgg16_bn.features,
        }
    }
}

impl Module for Extractor {
    fn forward(&self, xs: &Tensor) -> Tensor {
        println!("starting extractor module");
        let mut x: Tensor = Tensor::copy(xs);
        let mut out = vec![];


        let features = self.features.get_layers();

        println!("features: {:?}", features.len());

        for feature in features {
            x = feature.forward(&x);
            println!("x {:?}",&x);
            if xs.size()[2] <= 32 {
                out.push(Tensor::copy(&x));
            }
        }
        // convert out to tensor
        Tensor::cat(&out[1..], 1)
    }
}

#[derive(Debug)]
pub struct EAST {
    extractor: Extractor,
    merge: Merge,
    output: Output,
}

impl EAST {
    pub fn new<'p, P>(p: P) -> EAST
    where
        P: Borrow<tch::nn::Path<'p>>,
    {
        let extractor = Extractor::new(p.borrow());
        let merge = Merge::new(p.borrow());
        let output = Output::new(p.borrow(), 512);

        EAST {
            extractor,
            merge,
            output,
        }
    }
}

impl Module for EAST {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.output
            .forward(&self.merge.forward(&self.extractor.forward(xs)))
    }
}
