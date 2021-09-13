if [ ! -d "./results" ]; then
  mkdir ./results
fi

# MSCOCO Data
python train.py --data_path data/MS_COCO --maxlen 15 --save ./results/coco_result \
--batch_size 256 --niters_gan_d 1 --niters_gan_ae 1 --lr_gan_g 4e-04 --lr_ae 0.12 \
--add_noise --gan_d_local --enhance_dec

# WMTNews Data
python train.py --data_path data/NewsData --maxlen 32 --save ./results/news_result \
--batch_size 256 --niters_gan_d 1 --niters_gan_ae 1 --lr_gan_g 4e-04 --lr_ae 0.27 \
--add_noise --gan_d_local --enhance_dec