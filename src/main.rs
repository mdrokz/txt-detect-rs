use east_model::train::train;

fn main() {

    let train_img_path = "./ch4_training_images";
	let train_gt_path  = "./ch4_training_localization_transcription_gt";
	let pths_path      = "./pths";
	let batch_size     = 24; 
	let lr             = 1e-3;
	let num_workers    = 4;
	let epoch_iter     = 600;
	let save_interval  = 5;


    train(train_img_path.to_string(), train_gt_path.to_string(), pths_path.to_string(), batch_size, lr, num_workers, epoch_iter, save_interval);

}
