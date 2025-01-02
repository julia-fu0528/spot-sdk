# echo "################# COLLECT DATA ###########################"
# python store_robot_state.py 138.16.161.21 state \
#                             --output_dir data/gouger1209/stand_h5 \
#                             --markers_path data/gouger_markers_pos.txt\

# echo "################# DATALOADER ###############################"
# python dataset.py --session gouger1209 --data_dir data \
#                     --markers_path data/gouger_markers_pos.txt --classify --seq 3\

# echo "################# TRAINING ###############################"
# python train.py --session gouger1209 --data_dir data \
#                 --markers_path data/gouger_markers_pos.txt --device "gpu" --seq 3 --classify \


# echo "################# TRAINING ###############################"
# python train_tf.py --session gouger1209 --data_dir data \
#                 --model_dir model_torch --log_dir logs_torch --plots_dir plots_torch \
#                 --markers_path data/gouger_markers_pos.txt \

echo "################# PREDICTING ################################"
python predict.py 138.16.161.21 --ckpts_path gouger_logs/regression/version_170/checkpoints/last.ckpt\
                  --markers_path data/gouger_markers_pos.txt --data_dir data/gouger1209 --device cpu --seq 3\


# pip3 install torch torchvision torchaudio
# pip install pytorch_lightning==1.6.0
# pip install urdfpy
# pip install lightning
# pip install open3d
# pip install seaborn
# pip install -U 'tensorboard'




