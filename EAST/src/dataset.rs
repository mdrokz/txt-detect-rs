use std::f32::consts::PI;
use std::{fs, io};

use cv_convert::{FromCv, IntoCv, TchTensorAsImage, TryIntoCv};
use geo::{Area, BooleanOps, ConvexHull, Polygon};
use image::imageops::{brighten, contrast, crop, resize};
use image::{
    DynamicImage, GenericImage, GenericImageView, ImageBuffer, Pixel, Rgb, RgbImage, Rgba, SubImage,
};
use ndarray::{
    array, concatenate, s, Array, Array1, Array3, ArrayBase, ArrayView, Axis, Dim, IxDynImpl,
    OwnedRepr, Slice, SliceInfo, SliceInfoElem, ViewRepr,
};

use regex::Regex;

use opencv::core::{Mat, Scalar, ToInputArray, _InputArray};
use opencv::imgproc::fill_poly;

use rand::{random, thread_rng, Rng};
use tch::{Device, Kind, Tensor};

type Array2 = ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>;

// calculate the Euclidean distance
fn cal_distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    f32::sqrt((x1 - x2).powf(2.0) + (y1 - y2).powf(2.0))
    // math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
}

fn move_points(vertices: &mut Array1<f32>, index1: i64, index2: i64, r: &[f32; 4], coef: f32) {
    let index1 = index1 % 4;
    let index2 = index2 % 4;
    let x1_index = index1 * 2 + 0;
    let y1_index = index1 * 2 + 1;
    let x2_index = index2 * 2 + 0;
    let y2_index = index2 * 2 + 1;

    let r1 = r[index1 as usize];
    let r2 = r[index2 as usize];

    let length_x = vertices[x1_index as usize] - vertices[x2_index as usize];

    let length_y = vertices[y1_index as usize] - vertices[y2_index as usize];

    let length = cal_distance(
        vertices[x1_index as usize],
        vertices[y1_index as usize],
        vertices[x2_index as usize],
        vertices[y2_index as usize],
    );

    if length > 1.0 {
        let mut ratio = (r1 * coef) / length;
        vertices[x1_index as usize] = vertices[x1_index as usize] + length_x * ratio;
        vertices[y1_index as usize] = vertices[y1_index as usize] + length_y * ratio;

        ratio = (r2 * coef) / length;
        vertices[x2_index as usize] = vertices[x2_index as usize] - length_x * ratio;
        vertices[y2_index as usize] = vertices[y2_index as usize] - length_y * ratio;
    }
}

fn shrink_poly(vertices: &Array1<f32>, coef: f32) -> Array1<f32> {
    // println!("here {}", vertices);
    // extract x1,y1,x2,y2,x3,y3,x4,y4 from vertices
    let (x1, y1, x2, y2, x3, y3, x4, y4) = (
        vertices[0],
        vertices[1],
        vertices[2],
        vertices[3],
        vertices[4],
        vertices[6],
        vertices[6],
        vertices[7],
    );

    println!(
        "x1: {}, y1: {}, x2: {}, y2: {}, x3: {}, y3: {}, x4: {}, y4: {}",
        x1, y1, x2, y2, x3, y3, x4, y4
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

    // println!("just before move_points");

    move_points(&mut vertices, 0 + offset, 1 + offset, &r, coef);
    move_points(&mut vertices, 2 + offset, 3 + offset, &r, coef);
    move_points(&mut vertices, 1 + offset, 2 + offset, &r, coef);
    move_points(&mut vertices, 3 + offset, 4 + offset, &r, coef);

    vertices
}

fn get_rotate_mat(theta: f32) -> Array2 {
    array![[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]]
}

fn rotate_vertices<'a>(
    vertices: &Array1<f32>,
    theta: f32,
    anchor: Option<Array2>,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>> {
    let vertices = vertices.clone().into_shape((4, 2)).unwrap().t().to_owned();
    let anchor = anchor.map_or(array![[vertices[[0, 0]]], [vertices[[0, 1]]]], |f| f);

    let rotate_mat = get_rotate_mat(theta);

    let res = rotate_mat.dot(&(vertices - &anchor));

    // (res + anchor).T.reshape(-1) negative reshape
    let r = (res + anchor).t().to_owned();

    r.axis_iter(Axis(0))
        .map(|x| [x[0], x[1]])
        .flatten()
        .collect()
}

fn get_boundary(vertices: &Array1<f32>) -> (f32, f32, f32, f32) {
    let (x1, y1, x2, y2, x3, y3, x4, y4) = (
        vertices[0],
        vertices[1],
        vertices[2],
        vertices[3],
        vertices[4],
        vertices[5],
        vertices[6],
        vertices[7],
    );

    let x_min = x1.min(x2).min(x3).min(x4);
    let x_max = x1.max(x2).max(x3).max(x4);

    let y_min = y1.min(y2).min(y3).min(y4);

    let y_max = y1.max(y2).max(y3).max(y4);

    (x_min, x_max, y_min, y_max)
}

fn cal_error(vertices: Array1<f32>) -> f32 {
    let (x_min, x_max, y_min, y_max) = get_boundary(&vertices);
    let (x1, y1, x2, y2, x3, y3, x4, y4) = (
        vertices[0],
        vertices[1],
        vertices[2],
        vertices[3],
        vertices[4],
        vertices[5],
        vertices[6],
        vertices[7],
    );
    let err = cal_distance(x1, y1, x_min, y_min)
        + cal_distance(x2, y2, x_max, y_min)
        + cal_distance(x3, y3, x_max, y_max)
        + cal_distance(x4, y4, x_min, y_max);
    err
}

fn meshgrid_2d(
    coords_x: ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
    coords_y: ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
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
    anchor_x: f32,
    anchor_y: f32,
    length: f32,
) -> (
    ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
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

fn adjust_height(image: RgbImage, vertices: Array2, ratio: f32) -> (RgbImage, Array2) {
    // let ratio_h = 1.0 + ratio * (Array::random((), Uniform::new(0.0, 1.0)) * 2.0 - 1.0);
    let ratio_h = 1.0 + ratio * (random::<f32>() * 2.0 - 1.0);

    let old_h = image.height() as f32;

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

fn is_cross_text(
    start_loc: (f32, f32),
    length: f32,
    vertices: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
) -> bool {
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

    // println!("A {:?}", a);

    let points = a.axis_iter(Axis(0)).map(|x| (x[0], x[1])).collect();
    // convert a to rust vector

    let p1 = Polygon::new(points, vec![]).convex_hull();

    for vertice in vertices.outer_iter() {
        // println!("point vertice {:?}", vertice);
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
    vertices: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    labels: Array1<i32>,
    length: u32,
) -> (
    SubImage<&mut ImageBuffer<Rgb<u8>, Vec<u8>>>,
    ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
) {
    let (h, w) = image.dimensions();

    if h <= w && w < length {
        // let mut image = image.borrow_mut();
        *image = resize(
            &*image,
            length,
            h * length / w,
            image::imageops::FilterType::Gaussian,
        );
        // println!("Dimensions resize 1 {:?}", image.dimensions());
    } else if h < w && h < length {
        // let mut image = image.borrow_mut();
        *image = resize(
            &*image,
            w * length / h,
            length,
            image::imageops::FilterType::Gaussian,
        );
        // println!("Dimensions resize 2 {:?}", image.dimensions());
    }

    let ratio_h = image.height() / w;
    let ratio_w = image.width() / h;

    assert!(ratio_w >= 1 && ratio_h >= 1);

    // let new_vertices = Array2::zeros(vertices.shape());
    let mut new_vertices: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> =
        Array::zeros(vertices.shape());

    // let slice_w = Slice::new(0, Some(6), 2);

    // let slice_h = Slice::new(1, Some(7), 2);

    if vertices.len() > 0 {
        // new_vertices.slice_mut(s![0..6; 2,1]).assign(
        //     &vertices
        //         .slice_axis(Axis(1), slice_w)
        //         .mapv(|v| v * Into::<f32>::into(ratio_w)),
        // );
        // println!("new vertice {:?}", new_vertices.ndim());
        // println!("s dims {:?}", s![..,0..6; 2].in_ndim());
        new_vertices
            .slice_mut(s![..,0..6; 2])
            .assign(&vertices.slice(s![..,0..6; 2]).mapv(|v| v * ratio_w as f32));

        new_vertices
            .slice_mut(s![..,1..7; 2])
            .assign(&vertices.slice(s![..,1..7; 2]).mapv(|v| v * ratio_h as f32));

        // new_vertices.slice_mut(s![1..7; 2,1]).assign(
        //     &vertices
        //         .slice_axis(Axis(1), slice_h)
        //         .mapv(|v| v * Into::<f32>::into(ratio_h)),
        // );

        // new_vertices = vertices
        //     .slice_axis(Axis(1), slice_w)
        //     .mapv(|v| v * Into::<f32>::into(ratio_w));
        // new_vertices = vertices
        //     .slice_axis(Axis(1), slice_h)
        //     .mapv(|v| v * Into::<f32>::into(ratio_h));
    }

    // println!("new vertices: {:?}", new_vertices);

    let remain_h = image.height() - length;
    let remain_w = image.width() - length;

    let mut cnt = 0;

    let mut start_w = 0.0;
    let mut start_h = 0.0;

    loop {
        cnt += 1;

        start_w = random::<f32>() * remain_w as f32;
        start_h = random::<f32>() * remain_h as f32;

        // println!("start_w: {:?}", start_w);
        // println!("start_h: {:?}", start_h);

        // new_vertices[labels==1,:]

        // let mut text_vertices = new_vertices
        //     .slice(s![labels == 1, ..])

        let labels = labels.mapv(|f| (f == 1) as usize);

        // let vertice = v.select(Axis(0), &labels.as_slice().unwrap());

        // let x = 0..vertices.nrows();
        // println!("arrays: {:?}", vertices);

        let arrays: Vec<ArrayBase<ViewRepr<&f32>, Dim<IxDynImpl>>> = labels
            .indexed_iter()
            .filter(|(_, l)| **l == 1)
            .map(|(i, _l)| {
                // println!("index {} {}", i, _l);
                // println!("vertices: {:?}", vertices);
                vertices.slice_axis(Axis(0), Slice::new(i.try_into().unwrap(), None, 1))
            })
            .collect();

        // let vertice1 = ndarray::stack(Axis(0),&broadcasted_arrays).unwrap();
        // Array::co(Axis(0), &arrays);
        let vertice = ndarray::concatenate(Axis(0), &arrays).unwrap();

        // println!("concat vertice: {:?}", vertice);

        // println!("vertice1: {:?}", vertice1);
        if !is_cross_text((start_w, start_h), length as f32, vertice) && cnt > 1000 {
            break;
        }
    }

    // println!("img: {:?}", image.dimensions());

    // convert start_w to i32
    // let start_w = Into::<i32>::into(start_w);

    // println!("start_w: {}, start_h: {}", start_w as i32, start_h as i32);
    // println!("length: {}", length);

    let region = crop(
        image,
        start_w as u32,
        start_h as u32,
        start_w as u32 + length,
        start_h as u32 + length,
    );

    let len = new_vertices.len();

    if len == 0 {
        (region, new_vertices)
    } else {
        // println!("new vertices: {:?}", new_vertices);
        let cloned_v = new_vertices.clone();
        new_vertices.slice_mut(s![..,0..6; 2]).assign(
            &cloned_v
                .slice(s![..,0..6; 2])
                .mapv(|v| v - Into::<f32>::into(start_w)),
        );
        new_vertices.slice_mut(s![..,1..7; 2]).assign(
            &cloned_v
                .slice(s![..,1..7; 2])
                .mapv(|v| v - Into::<f32>::into(start_h)),
        );
        // new_vertices = new_vertices
        //     .slice_axis(Axis(1), slice_w)
        //     .mapv(|v| v - Into::<f32>::into(start_w))
        //     .to_owned();
        // new_vertices = new_vertices
        //     .slice_axis(Axis(1), slice_h)
        //     .mapv(|v| v - Into::<f32>::into(start_h))
        //     .to_owned();
        // new_vertices[[0, 0]] -= start_w as f32;
        // new_vertices[[1, 0]] -= start_h as f32;
        // new_vertices[[2, 0]] -= start_w as f32;
        // new_vertices[[3, 0]] -= start_h as f32;
        // new_vertices[[4, 0]] -= start_w as f32;
        // new_vertices[[5, 0]] -= start_h as f32;
        // new_vertices[[6, 0]] -= start_w as f32;
        // new_vertices[[7, 0]] -= start_h as f32;

        (region, new_vertices)
    }
}

fn rotate_image(
    image: &mut RgbImage,
    vertices: Array2,
    angle_range: f32,
) -> (&mut RgbImage, ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>) {
    let (h, w) = image.dimensions();

    let center_x = (w - 1) as f32 / 2.0;
    let center_y = (h - 1) as f32 / 2.0;

    let angle = angle_range * (random::<f32>() * 2.0 - 1.0);

    let angle = angle % 360.0;

    if angle == 180.0 {
        *image = image::imageops::rotate180(&*image);
    } else if (angle > 90.0 && angle < 270.0) && (h == w) {
        if angle == 90.0 {
            *image = image::imageops::rotate90(&*image);
        } else {
            *image = image::imageops::rotate270(&*image);
        }
    }

    let mut new_vertices: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> =
        Array::zeros(vertices.shape());

    // println!("new_vertices: {:?}", vertices);

    for (i, vertice) in vertices.outer_iter().enumerate() {
        // println!("vertice: {:?}", vertice);
        // println!("vertice: {:?}", vertice.to_shape((4,2)));
        new_vertices.slice_mut(s![i, ..]).assign(&rotate_vertices(
            &vertice.to_owned(),
            -angle / 180.0 * PI,
            Some(array![[center_x], [center_y]]),
        ))
    }

    (image, new_vertices)
}

fn find_min_rect_angle(vertices: &Array1<f32>) -> f32 {
    let angle_interval = 1;
    let angle_list = (0..90).step_by(angle_interval).collect::<Vec<_>>();
    let mut area_list = vec![];

    for theta in &angle_list {
        let rotated = rotate_vertices(vertices, *theta as f32 / 180.0 * std::f32::consts::PI, None);
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

    // sort area_list
    area_list.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sorted_area = (0..area_list.len())
        .map(|i| (i, area_list[i]))
        .collect::<Vec<_>>();

    sorted_area.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let sorted_area_index = sorted_area
        .iter()
        .map(|(i, _)| i.clone())
        .collect::<Vec<_>>();

    let mut min_error = f32::INFINITY;
    let mut best_index: i64 = -1;
    let rank_num = 10;

    for index in sorted_area_index[..rank_num].iter() {
        let rotated = rotate_vertices(
            vertices,
            angle_list[*index] as f32 / 180.0 * std::f32::consts::PI,
            None,
        );
        let temp_error = cal_error(rotated);

        if temp_error < min_error {
            min_error = temp_error;
            best_index = *index as i64;
        }
    }

    if best_index == -1 {
        angle_list[angle_list.len() - 1] as f32 / 180.0 * std::f32::consts::PI
    } else {
        angle_list[best_index as usize] as f32 / 180.0 * std::f32::consts::PI
    }
}

fn extract_vertices(lines: Vec<String>) -> (Array2, Array1<i32>) {
    let mut vertices = vec![];
    let mut labels = vec![];

    let re = Regex::new(r"[^\d]").unwrap();

    for line in lines {
        let mut v = vec![];
        for s in line.split(',').take(8) {
            let t = re.replace_all(s, "");

            if !s.is_empty() {
                match t.parse::<i64>() {
                    Ok(i) => v.push(i as f32),
                    Err(_) => (),
                }
            }
        }
        if !v.is_empty() {
            vertices.push(v);
        }
        let label = if line.contains("###") { 0 } else { 1 };
        labels.push(label);
    }

    // println!("vertices: {:?}", vertices);

    (
        Array2::from_shape_vec(
            (vertices.len(), 8),
            vertices.into_iter().flatten().collect(),
        )
        .unwrap(),
        Array::from(labels),
    )
}

fn index_array(
    rows: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    cols: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    a: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let mut b = Array::zeros((rows.shape()[0], cols.shape()[1]));
    // for (i, row) in cols.iter().enumerate() {
    //     for (j, col) in rows.iter().enumerate() {
    //         b[[i, j]] = a[[*row as usize, *col as usize]];
    //     }
    // }

    for (i, row) in rows.axis_iter(Axis(0)).enumerate() {
        for (j, col) in cols.axis_iter(Axis(1)).enumerate() {
            // println!("x: {}", x);
            // println!("i j {} {}", i, j);
            b[(i, j)] = a[[row[0] as usize, col[0] as usize]];
        }
    }

    b
}

#[repr(transparent)]
struct MatWrapper(Mat);

impl ToInputArray for MatWrapper {
    fn input_array(&self) -> std::result::Result<_InputArray, opencv::Error> {
        Ok(self.0.input_array().unwrap())
    }
}

// rust version
fn get_score_geo(
    img: &RgbImage,
    vertices: &Array2,
    labels: &Array1<i32>,
    scale: f32,
    length: i64,
) -> (Tensor, Tensor, Tensor) {
    // println!("scale {}", img.height());
    // println!("dimensions {}", (img.height() as f32 * scale) as usize);
    let mut score_map: Array3<f32> = Array3::zeros((
        // (img.height() as f32 * scale) as usize,
        // (img.width() as f32 * scale) as usize,
        128, 128, 1,
    ));
    let mut geo_map: Array3<f32> = Array3::zeros((
        // (img.height() as f32 * scale) as usize,
        // (img.width() as f32 * scale) as usize,
        128, 128, 5,
    ));
    let mut ignored_map: Array3<f32> = Array3::zeros((
        // (img.height() as f32 * scale) as usize,
        // (img.width() as f32 * scale) as usize,
        128, 128, 1,
    ));

    let index = Array::range(0., length as f32, 1. / scale);
    let (index_x, index_y) = meshgrid_2d(index.clone(), index);

    let mut ignored_polys = vec![];
    let mut polys = vec![];

    for i in 0..vertices.nrows() {
        let vertice = vertices.row(i).to_owned();

        // println!("vertice nrows: {:?}", vertice);

        if labels[i] == 0 {
            let mut v = scale * vertice.into_shape((4, 2)).unwrap().to_owned();
            v.mapv_inplace(|x| x.round());
            ignored_polys.push(v.mapv(|x| x as i32));
            continue;
        }

        // println!("scale: {}", scale);
        // println!("vertice: {:?}", vertice);
        let poly = (scale
            * shrink_poly(&vertice, 0.3)
                .into_shape((4, 2))
                .unwrap()
                .to_owned())
        .mapv(|x| {
            // println!("{}",x);
            x as i32
        });

        // println!("poly nrows: {:?}", poly);

        polys.push(poly.clone());

        let shape = score_map.shape();
        // println!("shape: {:?}", shape);
        // let shape: Vec<usize> = shape[..shape.len() - 1].into();

        // let temp_mask: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> = Array::zeros(shape);

        let shape: Vec<usize> = score_map.shape()[..2].into();
        // println!("shape: {:?}", shape);
        let temp_mask: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> = Array::zeros(shape);

        // score_map.shape[:-1]
        // let temp_mask: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> =
        //     Array::zeros((score_map.shape()[0], score_map.shape()[1]));

        // let temp_slice: Vec<Vec<f32>> = temp_mask
        //     .axis_iter(Axis(0))
        //     .map(|x| vec![x[0], x[1]])
        //     .collect();
        let temp_slice: Vec<Vec<f32>> = temp_mask
            .outer_iter()
            .map(|row| row.iter().map(|x| x.to_owned()).collect::<Vec<f32>>())
            .collect();
        // let temp_slice: Vec<Vec<f32>> = temp_mask.iter().map(|&x| vec![x]).collect();

        // temp_mask to mat
        let mut temp_mat = Mat::from_slice_2d(&temp_slice).unwrap();

        let poly_slice: Vec<Vec<i32>> = poly
            .outer_iter()
            .map(|row| row.iter().map(|x| x.to_owned()).collect::<Vec<i32>>())
            .collect();

        // println!("poly_slice: {:?}", poly_slice);

        let poly_mat = Mat::from_slice_2d(&poly_slice).unwrap();

        // color 1
        let color = Scalar::new(1., 1., 1., 1.);

        // println!("temp_mat: {:?}", temp_slice);

        fill_poly(
            &mut temp_mat,
            &poly_mat,
            color,
            0,
            0,
            opencv::core::Point_::default(),
        )
        .unwrap();

        let theta = find_min_rect_angle(&vertice);
        let rotate_mat = get_rotate_mat(theta);

        // vertice first index
        // let vertice = vertice[0][0];
        let anchor_x = vertice[0];
        let anchor_y = vertice[1];
        // let anchor_x = vertices.row(i)[0];
        // let anchor_y = vertices.row(i)[1];

        // println!("end fill_poly");

        let rotated_vertices = rotate_vertices(&vertice, theta, None);
        let (x_min, x_max, y_min, y_max) = get_boundary(&rotated_vertices);
        let (rotated_x, rotated_y) =
            rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length as f32);

        // println!("end rotate_all_pixels");

        let d1 = rotated_y.clone() - y_min;

        let d1_mask = d1.mapv(|x| if x < 0. { 1. } else { 0. });

        let d2 = y_max - rotated_y;

        let d2_mask = d2.mapv(|x| if x < 0. { 1. } else { 0. });

        let d3 = rotated_x.clone() - x_min;

        let d3_mask = d3.mapv(|x| if x < 0. { 1. } else { 0. });

        let d4 = x_max - rotated_x;

        let d4_mask = d4.mapv(|x| if x < 0. { 1. } else { 0. });

        // println!("end masking");

        // convert geo_map[:,:,0] += d1[index_y, index_x] * temp_mask
        // index d1 by index_y, index_x

        // geo_map[[0, 0, 0]] += d1[[index_y, index_x]] * temp_mask

        // println!("{:?} mat", temp_mat);

        let temp_tensor: ArrayView<'_, f32, Dim<IxDynImpl>> = (&temp_mat).try_into_cv().unwrap();

        let temp_tensor = temp_tensor.into_shape((128, 128)).unwrap().to_owned();

        // println!("MATT {:?}", temp_tensor);

        // let temp_tensor: ArrayView<'_, f32, Dim<[usize; 2]>> = (&temp_mat).try_into_cv().unwrap();
        // println!("temp_tensor {:?}", temp_tensor.shape());
        // println!(
        //     "indexed array d1 {:?}",
        //     index_array(index_y.clone(), index_x.clone(), d1_mask.clone()).mapv(|f| f as f32)
        //         * temp_tensor.clone()
        // );
        // println!(
        //     "indexed array d2 {:?}",
        //     index_array(index_y.clone(), index_x.clone(), d2_mask.clone())
        // );
        // println!(
        //     "indexed array d3 {:?}",
        //     index_array(index_y.clone(), index_x.clone(), d3_mask.clone())
        // );
        // println!(
        //     "indexed array d4 {:?}",
        //     index_array(index_y.clone(), index_x.clone(), d4_mask.clone())
        // );

        let g = geo_map.clone();

        geo_map.slice_mut(s![.., .., 0]).assign(
            &(g.slice(s![.., .., 0]).to_owned()
                + index_array(index_y.clone(), index_x.clone(), d1_mask).mapv(|f| f as f32)
                    * temp_tensor.clone()),
        );

        geo_map.slice_mut(s![.., .., 1]).assign(
            &(g.slice(s![.., .., 1]).to_owned()
                + index_array(index_y.clone(), index_x.clone(), d2_mask).mapv(|f| f as f32)
                    * temp_tensor.clone()),
        );

        geo_map.slice_mut(s![.., .., 2]).assign(
            &(g.slice(s![.., .., 2]).to_owned()
                + index_array(index_y.clone(), index_x.clone(), d3_mask).mapv(|f| f as f32)
                    * temp_tensor.clone()),
        );

        geo_map.slice_mut(s![.., .., 3]).assign(
            &(g.slice(s![.., .., 3]).to_owned()
                + index_array(index_y.clone(), index_x.clone(), d4_mask).mapv(|f| f as f32)
                    * temp_tensor.clone()),
        );

        geo_map
            .slice_mut(s![.., .., 4])
            .assign(&(g.slice(s![.., .., 4]).to_owned() + theta as f32 * temp_mask));
    }

    println!("geo map assignment complete");

    // let row: i32 = ignored_map.shape()[0].try_into().unwrap();

    // size of column
    // let col: i32 = ignored_map.shape()[1].try_into().unwrap();

    let (ignored_map_rows, ignored_map_cols, ignored_map_channels) = ignored_map.dim();

    let ignored_typ = opencv::core::CV_MAKETYPE(opencv::core::CV_32S, ignored_map_channels as i32);

    let (score_map_rows, score_map_cols, score_map_channels) = score_map.dim();

    let score_typ = opencv::core::CV_MAKETYPE(opencv::core::CV_32S, score_map_channels as i32);

    unsafe {
        let mut ignored_map_mat = Mat::new_rows_cols_with_data(
            ignored_map_rows as i32,
            ignored_map_cols as i32,
            ignored_typ,
            ignored_map.as_ptr() as *mut std::ffi::c_void,
            opencv::core::Mat_AUTO_STEP,
        )
        .unwrap();

        // println!("ignored map mat {:?}", ignored_map_mat);

        // let ignored_poly_slice: Vec<Vec<i32>> = ignored_polys
        //     .iter()
        //     .map(|x| x.iter().map(|y| y.to_owned()).collect::<Vec<i32>>())
        //     .collect();

        let ignored_poly_slice: Vec<Vec<i32>> = ignored_polys
            .iter()
            .map(|x| {
                x.to_owned()
                    .outer_iter()
                    .map(|row| row.iter().map(|x| x.to_owned()).collect::<Vec<i32>>())
                    .collect::<Vec<Vec<i32>>>()
            })
            .flatten()
            .collect();

        // let ignored_polys_mat = Mat::new_rows_cols_with_data(
        //     ignored_polys.len() as i32,
        //     1,
        //     typ,
        //     ignored_polys.as_ptr() as *mut _,
        //     opencv::core::Mat_AUTO_STEP,
        // )
        // .unwrap();

        // println!("ignored poly slice {:?}", ignored_poly_slice.len());

        let ignored_polys_mat = opencv::core::Mat::from_slice_2d(&ignored_poly_slice).unwrap();
        // let ignored_polys_mat = opencv::core::Mat::from_slice(&ignored_poly_slice).unwrap();

        // println!("ignored polys mat {:?}", ignored_polys_mat);

        // opencv::core::vconcat(&MatWrapper(ignored_polys_mat), dst);

        fill_poly(
            &mut ignored_map_mat,
            &ignored_polys_mat,
            Scalar::new(1., 1., 1., 1.),
            0,
            0,
            opencv::core::Point_::default(),
        )
        .unwrap();

        println!("end ignored poly");

        let mut score_mat = Mat::new_rows_cols_with_data(
            score_map_rows as i32,
            score_map_cols as i32,
            score_typ,
            score_map.as_ptr() as *mut _,
            opencv::core::Mat_AUTO_STEP,
        )
        .unwrap();

        let polys_slice: Vec<Vec<i32>> = polys
            .iter()
            .map(|x| {
                x.to_owned()
                    .outer_iter()
                    .map(|row| row.iter().map(|x| x.to_owned()).collect::<Vec<i32>>())
                    .collect::<Vec<Vec<i32>>>()
            })
            .flatten()
            .collect();

        let polys_mat = opencv::core::Mat::from_slice_2d(&polys_slice).unwrap();

        fill_poly(
            &mut score_mat,
            &polys_mat,
            Scalar::new(1., 1., 1., 1.),
            0,
            0,
            opencv::core::Point_::default(),
        )
        .unwrap();

        let geo_tensor = Tensor::from_cv(&geo_map).permute(&[2, 0, 1]);

        let ignored_tensor: Tensor = ignored_map_mat.try_into_cv().unwrap();

        let ignored_tensor = ignored_tensor.permute(&[2, 0, 1]);

        let score_tensor: Tensor = score_mat.try_into_cv().unwrap();

        let score_tensor = score_tensor.permute(&[2, 0, 1]);

        (geo_tensor, ignored_tensor, score_tensor)
    }
}

#[derive(Debug, Clone, Default)]
pub struct DataSet {
    img_files: Vec<String>,
    gt_files: Vec<String>,
    scale: f32,
    length: u32,
}

impl DataSet {
    pub fn new(img_dir: String, gt_dir: String, scale: f32, length: u32) -> Self {
        let img_files = fs::read_dir(img_dir)
            .unwrap()
            .map(|res| res.map(|e| e.path().to_str().unwrap().to_string()))
            .collect::<Result<Vec<_>, io::Error>>()
            .unwrap();

        let gt_files = fs::read_dir(gt_dir)
            .unwrap()
            .map(|res| res.map(|e| e.path().to_str().unwrap().to_string()))
            .collect::<Result<Vec<_>, io::Error>>()
            .unwrap();

        DataSet {
            img_files,
            gt_files,
            scale,
            length,
        }
    }
}

fn saturation(input: ImageBuffer<Rgb<u8>, Vec<u8>>, factor: f32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = input.dimensions();
    let mut output = DynamicImage::new_rgb8(width, height);
    for x in 0..width {
        for y in 0..height {
            let pixel = input.get_pixel(x, y);
            let mut r = pixel.0[0] as f32 / 255.0;
            let mut g = pixel.0[1] as f32 / 255.0;
            let mut b = pixel.0[2] as f32 / 255.0;
            // let mut a = pixel.0[3] as f32 / 255.0;
            let mut a = 255.0;
            let l = 0.2126 * r + 0.7152 * g + 0.0722 * b + 0.0 * a;
            r = l + factor * (r - l);
            g = l + factor * (g - l);
            b = l + factor * (b - l);
            a = l + factor * (a - l);
            r = r.min(1.0).max(0.0);
            g = g.min(1.0).max(0.0);
            b = b.min(1.0).max(0.0);
            a = a.min(1.0).max(0.0);
            let pixel = image::Rgba([
                (r * 255.0) as u8,
                (g * 255.0) as u8,
                (b * 255.0) as u8,
                (a * 255.0) as u8,
            ]);
            output.put_pixel(x, y, pixel);
        }
    }
    output.into_rgb8()
}

fn normalize(input: &Tensor, mean: &Tensor, std: &Tensor) -> Tensor {
    // Normalize the input tensor
    input.subtract(mean).divide(std)
}

fn image_to_tensor(image: &RgbImage) -> Tensor {
    let (width, height) = image.dimensions();
    let data: Vec<u8> = image.to_vec();
    let channels = 3;
    let tensor_shape = [24, channels as i64, height as i64, width as i64];

    let tensor1 = Tensor::zeros(&tensor_shape, (Kind::Float, Device::cuda_if_available()));

    for i in 0..24 {
        let tensor = Tensor::of_slice(data.as_slice()).view([
            channels as i64,
            height as i64,
            width as i64,
        ]);
        tensor1
            .get(i)
            .copy_(&tensor);
    }

    // Create a Tensor in the tch crate using the pixel data and shape
    // let tensor = Tensor::of_slice(data.as_slice()).view(tensor_shape);
    // let mut tensor = Tensor::new(
    //     &[height as i64, width as i64, 3],
    //     (Device::Cpu, nn::Kind::Float),
    // );
    // for (x, y, pixel) in image.enumerate_pixels() {
    //     let r = pixel[0] as f32 / 255.0;
    //     let g = pixel[1] as f32 / 255.0;
    //     let b = pixel[2] as f32 / 255.0;
    //     tensor[(x as i64, y as i64, 0)] = r;
    //     tensor[(x as i64, y as i64, 1)] = g;
    //     tensor[(x as i64, y as i64, 2)] = b;
    // }
    tensor1
}

impl Iterator for DataSet {
    type Item = (Tensor, Tensor, Tensor, Tensor);

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.img_files.len()
    }

    fn next(&mut self) -> Option<Self::Item> {
        // iterate over the dataset and return the next image
        let gt_file = self.gt_files.iter().next().unwrap();

        let img_file = self.img_files.iter().next().unwrap();

        let gt_lines = fs::read_to_string(gt_file).unwrap();

        let gt_lines: Vec<String> = gt_lines.split('\r').map(|s| s.to_string()).collect();

        let (vertices, labels) = extract_vertices(gt_lines);

        let img = image::open(img_file).unwrap();

        // println!("DIMENSIONS 1 {:?}", img.dimensions());

        // get rgb image from the image
        let img = img.into_rgb8();

        println!("img tensor {:?}", image_to_tensor(&img));

        // println!("DIMENSIONS 2 {:?}", img.dimensions());

        let (mut img, vertices) = adjust_height(img, vertices, 0.2);

        // println!("DIMENSIONS adjust {:?}", img.dimensions());

        let (mut img, vertices) = rotate_image(&mut img, vertices, 10.0);

        // println!("DIMENSIONS rotate {:?}", img.dimensions());
        // println!("rotate tensor {:?}",image_to_tensor(img));

        // println!("vertices: {:?}", vertices);

        let (img, vertices) = crop_image(&mut img, vertices, labels.clone(), self.length);

        // println!("DIMENSIONS crop {:?}", img.dimensions());

        let color_jitter = |input: &RgbImage| -> RgbImage {
            let mut rng = thread_rng();
            let brightness = (rng.gen_range(0.5..1.5) * 100.0) as i32;
            let contrast = rng.gen_range(0.5..1.5);
            let sat = rng.gen_range(0.5..1.5);
            let hue = (rng.gen_range(-0.25..0.25) * 360.0) as i32;

            // Adjust the hue, contrast, brightness, and saturation of the input image
            let hue_image = image::imageops::huerotate(input, hue);
            let contrast_image = image::imageops::contrast(&hue_image, contrast);
            let brightness_image = image::imageops::brighten(&contrast_image, brightness);
            let jittered = saturation(brightness_image, sat);

            // println!("jittered {:?}", jittered);
            // let jittered = image::imageops::resize(&saturation_image, 256, 256, image::imageops::FilterType::Nearest);
            // Convert the jittered image back to a tensor
            jittered
        };

        let transform = |input: &RgbImage| -> Tensor {
            let jittered = color_jitter(input);
            // println!("jittered {:?}", jittered);
            // let tensor: Tensor = TchTensorAsImage::from_cv(&jittered).into_inner();
            let tensor = image_to_tensor(&jittered);
            // println!("tensor {}", tensor.totype(Kind::Float));
            println!("tensor 2 {:?}", image_to_tensor(input));
            println!(
                "tensor 3 {:?}",
                TchTensorAsImage::from_cv(input).into_inner()
            );
            // normalize(
            //     &tensor,
            //     &Tensor::of_slice(&[0.5, 0.5, 0.5]),
            //     &Tensor::of_slice(&[0.5, 0.5, 0.5]),
            // )
            tensor.totype(Kind::Float)
        };

        // println!("DIMENSIONS {:?}", img.dimensions());

        let img = img.to_image();

        // convert vertices to 2,2

        // println!("{:?}", vertices);

        // println!("{:?}", vertices);
        let vertices = vertices.into_shape((10, 8)).to_owned().unwrap();

        let (score_map, geo_map, ignored_map) =
            get_score_geo(&img, &vertices, &labels, self.scale, self.length.into());

        Some((transform(&img), score_map, geo_map, ignored_map))
    }
}
