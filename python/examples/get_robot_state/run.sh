# echo "################# COLLECT DATA ###########################"
# python store_robot_state.py 138.16.161.21 state \
#                             --output_dir data/gouger1209/stand_h0 \
#                             --markers_path data/gouger_markers_pos.txt\


# echo "################# TRAINING ###############################"
# python train.py --session gouger1209 --data_dir data \
#                 --model_dir model --log_dir logs --plots_dir plots\

echo "################# PREDICTING ################################"
python predict.py 138.16.161.21 --model_path model/gouger1209/best_model.keras\
                  --markers_path data/gouger_markers_pos.txt --data_dir data/gouger1209\