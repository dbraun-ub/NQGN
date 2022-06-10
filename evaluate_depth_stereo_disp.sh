model_name=NQGN_comp30
echo $model_name
python evaluate_depth_stereo_disp.py --data_path=/path/to/kitti2012 \
--model_name $model_name \
--load_weights_folder /path/to/model/weights \
--eval_mono --crit 0.07 --ref_crit 0.03 --eval_split stereo

model_name=NQGN_comp10
echo $model_name
echo crit 0.04
python evaluate_depth_stereo_disp.py --data_path=/path/to/kitti2012 \
--model_name $model_name \
--load_weights_folder /path/to/model/weights \
--eval_mono --crit 0.04 --ref_crit 0.014 --eval_split stereo
