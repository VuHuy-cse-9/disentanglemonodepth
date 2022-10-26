# Put monodepthv2 decoder and ADFA night encoder into same folder
python3 evaluate_depth.py --png \
                          --load_weights_folder /content/drive/MyDrive/Colab\ Notebooks/MinhHuy/MyProposeDisentangle/log/monodepthv2_day/models/weights_19 \
                          --data_path /content/drive/MyDrive/Colab\ Notebooks/Monodepth/Dataset/oxfordrobocar \
                          --eval_mono \
                          --eval_split oxford_day         