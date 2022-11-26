use geo::{Area, BooleanOps, ConvexHull, Polygon};
use image::imageops::{crop, resize};
use image::{ImageBuffer, Rgb, RgbImage, SubImage};
use ndarray::{
    array, concatenate, s, Array, Array1, ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr, ViewRepr,
};

use rand::random;

type Array2 = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;

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

    r.into_shape((8,)).unwrap()
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

fn adjust_height(
    image: RgbImage,
    vertices: Array2,
    ratio: f64,
) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, Array2) {
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

fn is_cross_text(start_loc: (f64, f64), length: f64, vertices: Array2) -> bool {
    if vertices.len() == 0 {
        return false;
    }

    let (start_w, start_h) = start_loc;

    let a = array![
        [start_w, start_h],
        [start_w + length, start_h],
        [start_w + length, start_h + length],
        [start_w, start_h + length]
    ]
    .into_shape((4, 2))
    .unwrap();

    let points = a.axis_iter(Axis(0)).map(|x| (x[0], x[1])).collect();
    // convert a to rust vector

    let p1 = Polygon::new(points, vec![]).convex_hull();

    for vertice in vertices.outer_iter() {
        let points = vertice
            .into_shape((4, 2))
            .unwrap()
            .axis_iter(Axis(0))
            .map(|x| (x[0], x[1]))
            .collect();

        let p2 = Polygon::new(points, vec![]).convex_hull();

        let inter = p1.intersection(&p2).signed_area();

        if 0.01 <= inter / p2.signed_area() && inter / p2.signed_area() <= 0.99 {
            return true;
        }
    }

    false
}

fn crop_image(
    image: &mut RgbImage,
    vertices: Array2,
    labels: Array2,
    length: u32,
) -> (
    SubImage<&mut ImageBuffer<Rgb<u8>, Vec<u8>>>,
    ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>,
) {
    let (h, w) = image.dimensions();

    if h <= w && w < length {
        *image = resize(
            &*image,
            length as u32,
            h * length / w,
            image::imageops::FilterType::Gaussian,
        );
    } else if h < w && h < length {
        *image = resize(
            &*image,
            w * length / h,
            length as u32,
            image::imageops::FilterType::Gaussian,
        );
    }

    let ratio_h = image.height() / w;
    let ratio_w = image.width() / h;

    assert!(ratio_w >= 1 && ratio_h >= 1);

    // let new_vertices = Array2::zeros(vertices.shape());
    let mut new_vertices: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>> =
        Array::zeros(vertices.shape());

    if vertices.len() > 0 {
        new_vertices[[0, 0]] = vertices[[0, 0]] * ratio_w as f64;
        new_vertices[[1, 0]] = vertices[[1, 0]] * ratio_h as f64;
        new_vertices[[2, 0]] = vertices[[2, 0]] * ratio_w as f64;
        new_vertices[[3, 0]] = vertices[[3, 0]] * ratio_h as f64;
        new_vertices[[4, 0]] = vertices[[4, 0]] * ratio_w as f64;
        new_vertices[[5, 0]] = vertices[[5, 0]] * ratio_h as f64;
        new_vertices[[6, 0]] = vertices[[6, 0]] * ratio_w as f64;
        new_vertices[[7, 0]] = vertices[[7, 0]] * ratio_h as f64;
    }

    let remain_h = image.height() - length;
    let remain_w = image.width() - length;

    let mut cnt = 0;

    let mut start_w = 0;
    let mut start_h = 0;

    loop {
        cnt += 1;

        start_w = random::<u32>() * remain_w;
        start_h = random::<u32>() * remain_h;

        // new_vertices[labels==1,:]

        // let mut text_vertices = new_vertices
        //     .slice(s![labels == 1, ..])
        let v = new_vertices.clone().into_shape((2, 2)).unwrap();

        if !is_cross_text((start_w.into(), start_h.into()), length.into(), v) && cnt > 1000 {
            break;
        }
    }

    let region = crop(image, start_w, start_h, start_w + length, start_h + length);

    if new_vertices.len() == 0 {
        (region, new_vertices)
    } else {
        new_vertices[[0, 0]] -= start_w as f64;
        new_vertices[[1, 0]] -= start_h as f64;
        new_vertices[[2, 0]] -= start_w as f64;
        new_vertices[[3, 0]] -= start_h as f64;
        new_vertices[[4, 0]] -= start_w as f64;
        new_vertices[[5, 0]] -= start_h as f64;
        new_vertices[[6, 0]] -= start_w as f64;
        new_vertices[[7, 0]] -= start_h as f64;

        (region, new_vertices)
    }
}

fn find_min_rect_angle(vertices: Array2) -> f64 {
    let angle_interval = 1;
    let angle_list = (0..90).step_by(angle_interval).collect::<Vec<_>>();
    let mut area_list = vec![];

    for theta in angle_list {
        let rotated = rotate_vertices(vertices, theta as f64 / 180.0 * std::f64::consts::PI, None);
        let x1 = rotated[0];
        let y1 = rotated[1];
        let x2 = rotated[2];
        let y2 = rotated[3];
        let x3 = rotated[4];
        let y3 = rotated[5];
        let x4 = rotated[6];
        let y4 = rotated[7];

        let temp_area = (x1.max(x2).max(x3).max(x4) - x1.min(x2).min(x3).min(x4))
            * (y1.max(y2).max(y3).max(y4) - y1.min(y2).min(y3).min(y4));

        area_list.push(temp_area);
    }

    let sorted_area_index = (0..area_list.len())
        .map(|i| (i, area_list[i]))
        .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .collect::<Vec<_>>();

    let mut min_error = f64::INFINITY;
    let mut best_index: i64 = -1;
    let rank_num = 10;

    for index in sorted_area_index[..rank_num].iter() {
        let rotated = rotate_vertices(
            vertices,
            angle_list[*index] as f64 / 180.0 * std::f64::consts::PI,
            None,
        );
        let temp_error = cal_error(rotated);

        if temp_error < min_error {
            min_error = temp_error;
            best_index = *index;
        }
    }

    if best_index == -1 {
        angle_list[angle_list.len() - 1] as f64 / 180.0 * std::f64::consts::PI
    } else {
        angle_list[best_index as usize] as f64 / 180.0 * std::f64::consts::PI
    }

}

fn extract_vertices(lines: Vec<String>) -> (Array2, Array1<i32>) {
    let mut vertices = vec![];
    let mut labels = vec![];

    for line in lines {
        let mut v = vec![];
        for s in line.split(',').take(8) {
            v.push(s.parse::<i64>().unwrap() as f64);
        }
        vertices.push(v);
        let label = if line.contains("###") { 0 } else { 1 };
        labels.push(label);
    }

    (
        Array2::from_shape_vec(
            (vertices.len(), 8),
            vertices.into_iter().flatten().collect(),
        )
        .unwrap(),
        Array::from(labels),
    )
}
