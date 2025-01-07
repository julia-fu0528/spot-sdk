# echo "################# COLLECT DATA ###########################"
# python store_robot_state.py 138.16.161.22 state \
#                             --markers_path data/gouger_markers_pos.txt\
#                             --output_dir data/gouger1209/tnc_l3 \

# echo "################# DATALOADER ###############################"
# python dataset.py --session gouger1209 --data_dir data \
#                     --markers_path data/gouger_markers_pos.txt --seq 3\

echo "################# TRAINING ###############################"
python train.py --session gouger1209 --data_dir data \
                --markers_path data/gouger_markers_pos.txt --device "gpu" --seq 3 \


# echo "################# TRAINING ###############################"
# python train_tf.py --session gouger1209 --data_dir data \
#                 --model_dir model_torch --log_dir logs_torch --plots_dir plots_torch \
#                 --markers_path data/gouger_markers_pos.txt \

# echo "################# PREDICTING ################################"
# python predict.py 138.16.161.22 --ckpts_path gouger_logs/regression/version_183/checkpoints/best.ckpt\
#                   --markers_path data/gouger_markers_pos.txt --data_dir data/gouger1209 --device cpu --seq 3 \
#                   --choreography-filepaths choreo/step.txt choreo/trot.txt choreo/turn_2step.txt choreo/twerk.txt choreo/unstow.txt\






# nc 20 seconds all tusker




# pip3 install torch torchvision torchaudio
# pip install pytorch_lightning==1.6.0
# pip install urdfpy
# pip install lightning
# pip install open3d
# pip install seaborn
# pip install -U 'tensorboard'




