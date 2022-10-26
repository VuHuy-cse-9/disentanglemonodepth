python3 train.py --png \                   
                  --num_epochs 40 \
                  --split oxford_da_test \  #[oxford_da_test, oxford_da]
                  --discrimination \
                  --num_discriminator 3 \   
                  --discriminator_mode conv\
                  --batch_size 32 \
                  --model_name adfav2_test_3conv_imagepool_inorm_4xlr_4xlr\
                  --log_dir /content/drive/MyDrive/Colab\ Notebooks/MinhHuy/MyProposeDisentangle/log/ \
                  --data_path /content/drive/MyDrive/Colab\ Notebooks/MinhHuy/MyProposeDisentangle/Oxfordrobocar \
                  --learning_rate 4e-4 \
                  --G_learning_rate 4e-4 \
                  --num_workers 4 \
                  --save_frequency 5 \
                  --pool_size 50 \
                  #--load_day_weights_folder /content/drive/MyDrive/Colab\ Notebooks/MinhHuy/MyProposeDisentangle/log/monodepthv2_day/models/weights_19 \
                  #--models_to_load domain_classifier_0 domain_classifier_1 domain_classifier_2 night_encoder