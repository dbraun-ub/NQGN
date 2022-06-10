# Eigen split
model_name=NQGN_comp30
echo $model_name
python evaluate_depth.py --data_path=/path/to/kitti \
  --model_name $model_name \
  --load_weights_folder /path/to/model/weights \
  --eval_mono --crit 0.07 --eval_split eigen

model_name=NQGN_comp10
echo $model_name
python evaluate_depth.py --data_path=/path/to/kitti \
--model_name $model_name \
--load_weights_folder /path/to/model/weights \
--eval_mono --crit 0.04 --eval_split eigen
