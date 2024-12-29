# echo "################# COLLECT DATA ###########################"
# python store_robot_state.py 138.16.161.21 state \
#                             --output_dir data/gouger1209/stand_h5 \
#                             --markers_path data/gouger_markers_pos.txt\

echo "################# DATALOADER ###############################"
python dataset.py --session gouger1209 --data_dir data \
                    --markers_path data/gouger_markers_pos.txt \

# echo "################# TRAINING ###############################"
# python train_torch.py --session gouger1209 --data_dir data \
#                 --model_dir model_torch --log_dir logs_torch --plots_dir plots_torch \
#                 --markers_path data/gouger_markers_pos.txt \

# echo "################# PREDICTING ################################"
# python predict.py 138.16.161.21 --model_path model/gouger1209/best_model.keras\
#                   --markers_path data/gouger_markers_pos.txt --data_dir data/gouger1209\


# pip3 install torch torchvision torchaudio
# pip install pytorch_lightning==1.6.0
