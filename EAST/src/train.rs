use std::time::{Instant, SystemTime};

use crate::dataset::DataSet;
use crate::{loss::Loss, model::EAST};
use tch::{
    nn::{self, Module, OptimizerConfig},
    Device, Tensor,
};

pub fn train(
    train_img_path: String,
    train_gt_path: String,
    pths_path: String,
    batch_size: i64,
    lr: f64,
    num_workers: i32,
    epoch_iter: i64,
    interval: i64,
) {
    let file_num = std::fs::read_dir(train_img_path.clone()).unwrap().count();

    let mut vs = nn::VarStore::new(Device::cuda_if_available());

    let dataset = DataSet::new(train_img_path, train_gt_path, 0.25, 512);

    let mut model = EAST::new(vs.root());

    let criterion = Loss {};

    let mut opt = nn::Adam::default().build(&vs, lr).unwrap();

    for epoch in 0..epoch_iter {
        let mut epoch_loss = 0;
        // current time
        let epoch_time = std::time::Instant::now();

        for (i, (img, gt_score, gt_geo, ignored_map)) in dataset.clone().enumerate() {
            let (img, gt_score, gt_geo, ignored_map) = (
                img.to_device(Device::cuda_if_available()),
                gt_score.to_device(Device::cuda_if_available()),
                gt_geo.to_device(Device::cuda_if_available()),
                ignored_map.to_device(Device::cuda_if_available()),
            );
            println!("cast to device complete");
            let start_time = std::time::Instant::now();
            let tensor_parts = model.forward(&img).split(2, 1);
            let pred_score = &tensor_parts[0];
            let pred_geo = &tensor_parts[1];

            println!("pred_score is {:?}, pred_geo is {:?}", pred_score, pred_geo);

            // &Tensor to Tensor
            let pred_score = pred_score.to_kind(tch::Kind::Float);
            let pred_geo = pred_geo.to_kind(tch::Kind::Float);

            let loss = criterion.forward(&Tensor::cat(
                &[gt_score, pred_score, gt_geo, pred_geo, ignored_map],
                1,
            ));

            let x: i64 = loss.get(0).into();

            epoch_loss += x;

            opt.zero_grad();

            loss.backward();

            opt.step();

            println!("Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8}, batch_loss is {:.8}",
            epoch + 1, epoch_iter, i + 1, (file_num / batch_size as usize) as i32, start_time.elapsed().as_secs_f64(), loss);
        }

        println!(
            "epoch_loss is {:.8}, epoch_time is {:.8}",
            epoch_loss / (file_num / batch_size as usize) as i64,
            Instant::now().duration_since(epoch_time).as_secs_f64()
        );

        let time = SystemTime::now();
        println!("{:?}", time);
        let s = "=".repeat(50);
        println!("= {}", s);

        if (epoch + 1) % interval == 0 {
            vs.save(format!("{}/east_{}.pth", pths_path, epoch + 1))
                .unwrap();
        }
    }
}
