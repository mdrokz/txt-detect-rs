use std::borrow::Borrow;

use tch::{Tensor,nn::{Module,Path, BatchNormConfig}};


#[derive(Debug)]
pub struct MaxPool2d<'a>(pub &'a [i64; 2], pub &'a [i64; 2]);

#[derive(Debug)]
pub struct ReLU;

#[derive(Debug)]
pub struct Dropout;

impl Module for Dropout {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.dropout(0.5, true)
    }
}

impl Module for ReLU {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.relu()
    }
}

impl Module for MaxPool2d<'_> {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.max_pool2d(self.0, self.1, &[0, 0], &[1], false)
    }
}

#[derive(Debug)]
pub struct BatchNorm {
    config: BatchNormConfig,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub ws: Option<Tensor>,
    pub bs: Option<Tensor>,
    pub nd: usize,
}

fn batch_norm<'a, T: Borrow<Path<'a>>>(
    vs: T,
    nd: usize,
    out_dim: i64,
    config: BatchNormConfig,
) -> BatchNorm {
    let vs = vs.borrow();
    let (ws, bs) = if config.affine {
        let ws = vs.var("weight", &[out_dim], config.ws_init);
        let bs = vs.var("bias", &[out_dim], config.bs_init);
        (Some(ws), Some(bs))
    } else {
        (None, None)
    };
    BatchNorm {
        config,
        running_mean: vs.zeros_no_train("running_mean", &[out_dim]),
        running_var: vs.ones_no_train("running_var", &[out_dim]),
        ws,
        bs,
        nd,
    }
}

pub fn batch_norm2d<'a, T: Borrow<Path<'a>>>(
    vs: T,
    out_dim: i64,
    config: BatchNormConfig,
) -> BatchNorm {
    batch_norm(vs, 2, out_dim, config)
}

/// Applies Batch Normalization over a five dimension input.
///
/// The input shape is assumed to be (N, C, D, H, W). Normalization
/// is performed over the first batch dimension N.
pub fn batch_norm3d<'a, T: Borrow<Path<'a>>>(
    vs: T,
    out_dim: i64,
    config: BatchNormConfig,
) -> BatchNorm {
    batch_norm(vs, 3, out_dim, config)
}

impl Module for BatchNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let dim = xs.dim();
        if self.nd == 1 && dim != 2 && dim != 3 {
            panic!(
                "as nd={}, expected an input tensor with 2 or 3 dims, got {} ({:?})",
                self.nd,
                dim,
                xs.size()
            )
        }
        if self.nd > 1 && xs.dim() != self.nd + 2 {
            panic!(
                "as nd={}, expected an input tensor with {} dims, got {} ({:?})",
                self.nd,
                self.nd + 2,
                dim,
                xs.size()
            )
        };
        Tensor::batch_norm(
            xs,
            self.ws.as_ref(),
            self.bs.as_ref(),
            Some(&self.running_mean),
            Some(&self.running_var),
            true,
            self.config.momentum,
            self.config.eps,
            self.config.cudnn_enabled,
        )
    }
}


#[derive(Debug)]
pub struct AdaptiveAvgPool2d(pub 
    [i64; 2]);

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.adaptive_avg_pool2d(&self.0)
    }
}