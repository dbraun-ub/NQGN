python ../train.py --model_name train_model \
  --data_path=/path/to/kitti \
  --log_dir /path/to/model/save \
  --crit 0.025  \
  --use_stereo --frame_ids 0 --batch_size 6 \
  --load_weights_folder  /path/to/pretrained/model \
  --models_to_load encoder depth \
  --scales 0 1 2 3 4 5
