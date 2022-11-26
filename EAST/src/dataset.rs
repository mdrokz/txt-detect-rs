use image::imageops::resize;
use image::{RgbImage, ImageBuffer, Rgb};
// use image::{ImageBuffer, RgbImage};
use ndarray::{
    array, concatenate, s, Array, ArrayBase, Axis, Dim, Dimension, IxDynImpl, OwnedRepr, ViewRepr,
};

use rand::random;

type Array2 = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;

// pub fn round<A, D>(a: Array<A, D>) -> Array<A, D>
// where
//     A: Float,
//     D: Dimension,
// {
//     a.mapv(|x| x.round())
// }

// calculate the Euclidean distance
fn cal_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    f64::sqrt((x1 - x2).powf(2.0) + (y1 - y2).powf(2.0))
    // math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
}

fn move_points(vertices: &mut Array2, index1: i64, index2: i64, r: &[f64; 4], coef: f64) {
    let index1 = index1 % 4;
    let index2 = index2 % 4;
    let x1_index = index1 * 2 + 0;
    let y1_index = index1 * 2 + 1;
    let x2_index = index2 * 2 + 0;
    let y2_index = index2 * 2 + 1;

    let r1 = r[index1 as usize];
    let r2 = r[index2 as usize];

    let length_x = vertices[[x1_index as usize, 0]] - vertices[[x2_index as usize, 0]];

    let length_y = vertices[[y1_index as usize, 0]] - vertices[[y2_index as usize, 0]];

    let length = cal_distance(
        vertices[[x1_index as usize, 0]],
        vertices[[y1_index as usize, 0]],
        vertices[[x2_index as usize, 0]],
        vertices[[y2_index as usize, 0]],
    );

    if length > 1.0 {
        let mut ratio = (r1 * coef) / length;
        vertices[[x1_index as usize, 0]] = vertices[[x1_index as usize, 0]] + length_x * ratio;
        vertices[[y1_index as usize, 0]] = vertices[[y1_index as usize, 0]] + length_y * ratio;

        ratio = (r2 * coef) / length;
        vertices[[x2_index as usize, 0]] = vertices[[x2_index as usize, 0]] - length_x * ratio;
        vertices[[y2_index as usize, 0]] = vertices[[y2_index as usize, 0]] - length_y * ratio;
    }
}

fn shrink_poly(vertices: Array2, coef: f64) {
    // extract x1,y1,x2,y2,x3,y3,x4,y4 from vertices
    let (x1, y1, x2, y2, x3, y3, x4, y4) = (
        vertices[[0, 0]],
        vertices[[1, 0]],
        vertices[[2, 0]],
        vertices[[3, 0]],
        vertices[[4, 0]],
        vertices[[5, 0]],
        vertices[[6, 0]],
        vertices[[7, 0]],
    );

    let r1 = cal_distance(x1, y1, x2, y2).min(cal_distance(x1, y1, x4, y4));
    let r2 = cal_distance(x2, y2, x1, y1).min(cal_distance(x2, y2, x3, y3));
    let r3 = cal_distance(x3, y3, x2, y2).min(cal_distance(x3, y3, x4, y4));
    let r4 = cal_distance(x4, y4, x1, y1).min(cal_distance(x4, y4, x3, y3));

    let mut offset = 0;

    let r = [r1, r2, r3, r4];

    if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4)
        > cal_distance(x2, y2, x3, y3) + cal_distance(x1, y1, x4, y4)
    {
        offset = 0;
    } else {
        offset = 1;
    }

    let mut vertices = vertices.to_owned();

    move_points(&mut vertices, 0 + offset, 1 + offset, &r, coef);
    move_points(&mut vertices, 2 + offset, 3 + offset, &r, coef);
    move_points(&mut vertices, 1 + offset, 2 + offset, &r, coef);
    move_points(&mut vertices, 3 + offset, 4 + offset, &r, coef);
}

fn get_rotate_mat(theta: f64) -> Array2 {
    array![[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]]
}

fn rotate_vertices<'a>(
    vertices: Array2,
    theta: f64,
    anchor: Option<Array2>,
) -> ArrayBase<ViewRepr<&'a f64>, Dim<[usize; 1]>> {
    let vertices = vertices.into_shape((4, 2)).unwrap().t().to_owned();
    let anchor = anchor.map_or(array![[vertices[[0, 0]]], [vertices[[0, 1]]]], |f| f);

    let rotate_mat = get_rotate_mat(theta);

    let res = rotate_mat.dot(&(vertices - &anchor));

    // (res + anchor).T.reshape(-1) negative reshape
    let r = (res + anchor).t();

    r.into_shape(1).unwrap()
}

fn get_boundary(vertices: &Array2) -> (f64, f64, f64, f64) {
    let (x1, y1, x2, y2, x3, y3, x4, y4) = (
        vertices[[0, 0]],
        vertices[[1, 0]],
        vertices[[2, 0]],
        vertices[[3, 0]],
        vertices[[4, 0]],
        vertices[[5, 0]],
        vertices[[6, 0]],
        vertices[[7, 0]],
    );

    let x_min = x1.min(x2).min(x3).min(x4);
    let x_max = x1.max(x2).max(x3).max(x4);

    let y_min = y1.min(y2).min(y3).min(y4);

    let y_max = y1.max(y2).max(y3).max(y4);

    (x_min, x_max, y_min, y_max)
}

fn cal_error(vertices: Array2) -> f64 {
    let (x_min, x_max, y_min, y_max) = get_boundary(&vertices);
    let (x1, y1, x2, y2, x3, y3, x4, y4) = (
        vertices[[0, 0]],
        vertices[[1, 0]],
        vertices[[2, 0]],
        vertices[[3, 0]],
        vertices[[4, 0]],
        vertices[[5, 0]],
        vertices[[6, 0]],
        vertices[[7, 0]],
    );
    let err = cal_distance(x1, y1, x_min, y_min)
        + cal_distance(x2, y2, x_max, y_min)
        + cal_distance(x3, y3, x_max, y_max)
        + cal_distance(x4, y4, x_min, y_max);
    err
}

fn meshgrid_2d(
    coords_x: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    coords_y: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
) -> (Array2, Array2) {
    let x_len = coords_x.shape()[0];
    let y_len = coords_y.shape()[0];

    let coords_x_s = coords_x.into_shape((1, y_len)).unwrap();
    let coords_x_b = coords_x_s.broadcast((x_len, y_len)).unwrap();
    let coords_y_s = coords_y.into_shape((x_len, 1)).unwrap();
    let coords_y_b = coords_y_s.broadcast((x_len, y_len)).unwrap();

    (coords_x_b.to_owned(), coords_y_b.to_owned())
}

fn rotate_all_pixels<'a>(
    rotate_mat: Array2,
    anchor_x: f64,
    anchor_y: f64,
    length: f64,
) -> (
    ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>,
    ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>,
) {
    let x = Array::range(0., length, 1.);
    let y = Array::range(0., length, 1.);

    // implement meshgrid in ndarray
    let (x, y) = meshgrid_2d(x, y);

    let xx = x.clone();
    let yy = y.clone();

    let x_len = x.len();
    let y_len = y.len();

    let x_shape = xx.shape();
    let y_shape = yy.shape();

    let x_lin = x.into_shape((1, x_len)).unwrap();
    let y_lin = y.into_shape((1, y_len)).unwrap();

    let coord_mat = concatenate![Axis(0), x_lin, y_lin];
    let coord_sum = coord_mat - array![[anchor_x], [anchor_y]];

    let rotated_coord = rotate_mat.dot(&coord_sum) + array![[anchor_x], [anchor_y]];

    let rotated_x = rotated_coord.slice(s![0, ..]).into_shape(x_shape).unwrap();

    let rotated_y = rotated_coord.slice(s![1, ..]).into_shape(y_shape).unwrap();

    (rotated_x.to_owned(), rotated_y.to_owned())
}

fn adjust_height(image: RgbImage, vertices: Array2, ratio: f64) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, Array2) {
    // let ratio_h = 1.0 + ratio * (Array::random((), Uniform::new(0.0, 1.0)) * 2.0 - 1.0);
    let ratio_h = 1.0 + ratio * (random::<f64>() * 2.0 - 1.0);

    let old_h = image.height() as f64;

    // implement around in ndarray
    // let new_h = round(old_h * ratio_h) as u32;
    let new_h = (old_h * ratio_h).round();

    let img = resize(
        &image,
        image.width(),
        new_h as u32,
        image::imageops::FilterType::Gaussian,
    );

    let mut new_vertices = vertices.to_owned();

    if vertices.len() > 0 {
        new_vertices[[1, 0]] = vertices[[1, 0]] * old_h;
        new_vertices[[3, 0]] = vertices[[3, 0]] * old_h;
        new_vertices[[5, 0]] = vertices[[5, 0]] * old_h;
        new_vertices[[7, 0]] = vertices[[7, 0]] * old_h;
    }

    (img, vertices)
}
