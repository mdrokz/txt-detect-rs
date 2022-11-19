use tch::{nn::Module, Tensor};

fn get_dice_loss(gt_score: &Tensor, pred_score: &Tensor) -> Tensor {
    // let inter = tch::
    let inter = Tensor::sum(&(gt_score * pred_score), tch::Kind::Uint8);
    let union = gt_score.sum(tch::Kind::Uint8) + pred_score.sum(tch::Kind::Uint8) + 1e-5;

    1. - (2 * inter / union)
}

fn get_geo_loss(gt_geo: &Tensor, pred_geo: &Tensor) -> (Tensor, Tensor) {
    let gt_split = gt_geo.split(1, 1);
    let pred_split = pred_geo.split(1, 1);

    let angle_gt = &gt_split[4];

    let angle_pred = &pred_split[4];

    let area_gt = (&gt_split[0] + &gt_split[1]) * (&gt_split[2] + &gt_split[3]);

    let area_pred = (&pred_split[0] + &pred_split[1]) * (&pred_split[2] + &pred_split[3]);

    let w_union = &gt_split[2].min_other(&pred_split[2]) + &gt_split[3].min_other(&pred_split[3]);

    let h_union = &gt_split[0].min_other(&pred_split[0]) + &gt_split[1].min_other(&pred_split[1]);

    let area_intersect = w_union * h_union;

    let area_union = area_gt + area_pred - &area_intersect;

    let iou_loss_map = -Tensor::log(&((area_intersect + 1.0) / (area_union + 1.0)));

    let angle_loss_map = 1 - Tensor::cos(&(angle_pred - angle_gt));

    (iou_loss_map, angle_loss_map)
}

#[derive(Debug)]
struct Loss {
    gt_score: Tensor,
    pred_score: Tensor,
    gt_geo: Tensor,
    pred_geo: Tensor,
    ignored_map: Tensor,
}

impl Loss {
    pub const WEIGHT_ANGLE: i32 = 10;
}

impl Module for Loss {
    fn forward(&self, _xs: &tch::Tensor) -> tch::Tensor {

        let gt_sum = f64::from(&self.gt_score.sum(tch::Kind::Float));

        if gt_sum < 1.0 {
            return Tensor::sum(&(&self.pred_score + &self.pred_geo), tch::Kind::Float) * 0;
        }

        let classify_loss = get_dice_loss(
            &self.gt_score,
            &(&self.pred_score * (1 - &self.ignored_map)),
        );

        let (iou_loss_map, angle_loss_map) = get_geo_loss(&self.gt_geo, &self.pred_geo);

        let angle_loss = Tensor::sum(&(&angle_loss_map * &self.gt_score), tch::Kind::Uint8)
            / Tensor::sum(&self.gt_score, tch::Kind::Uint8);

        let iou_loss = Tensor::sum(&(&iou_loss_map * &self.gt_score), tch::Kind::Uint8)
            / Tensor::sum(&self.gt_score, tch::Kind::Uint8);

        println!(
            "classify_loss: {:?}, angle_loss: {:?} iou loss {:?}",
            classify_loss, &angle_loss, &iou_loss
        );

        let geo_loss = Loss::WEIGHT_ANGLE * angle_loss + iou_loss;

        geo_loss + classify_loss
    }
}
