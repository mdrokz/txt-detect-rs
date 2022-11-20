// calculate the Euclidean distance
fn cal_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    f64::sqrt((x1 - x2).powf(2.0) + (y1 - y2).powf(2.0))
    // math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
}