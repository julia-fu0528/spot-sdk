echo "################# COLLECT DATA ###########################"
python store_robot_state.py 138.16.161.24 state \
                            --output_dir data/20241201 \
                            --markers_path data/markers_pos.txt\


# echo "################# TRAINING ###############################"
# python train.py --session 20241201 --data_dir data \
#                 --model_dir model --log_dir log --plots_dir plots\

# echo "################# PREDICTING ################################"
# python predict.py 138.16.161.24 --model_path model/20241201/best_model.keras\
#                   --markers_path data/markers_pos.txt --data_dir data/20241201\