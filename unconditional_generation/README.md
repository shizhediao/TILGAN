# Unconditional Generation Code

This is the implementaition for Unconditional Generation Task.

## Quick Links
- [Unconditional Generation Code](#unconditional-generation-code)
  - [Quick Links](#quick-links)
  - [Environment](#environment)
  - [Quick Start](#quick-start)
  - [Use with your own data](#use-with-your-own-data)
  - [Acknowledgement](#acknowledgement)

## Environment 
To run our code, please install all the dependency packages by using the following command under python 3.6 (recommend):

```
pip install -r requirements.txt
```

## Quick Start
After the environment setup, you can simply run the following command:
  ```shell
  CUDA_VISIBLE_DEVICES=<GPU_ID> bash train.sh
  ```

Or you can run the commends as following:

   ```shell
   mkdir results
   CUDA_VISIBLE_DEVICES=<GPU_ID> python train.py --data_path data/MS_COCO --maxlen 15 --save ./results/coco_result \
  --batch_size 256 --niters_gan_d 1 --niters_gan_ae 1 --lr_gan_g 4e-04 --lr_ae 0.12 \
  --add_noise --gan_d_local --enhance_dec
   ```
`GPU_ID=0/1/2/... the GPU you want to use`

You can simply add arguments `--add_noise` , `--gan_d_local` and `--enhance_dec` to test the variants of our model.

## Use with your own data
Our code allows you to do unconditional generation tasks, for example, generate fluent sentences from scratch.
In this section, we provide instructions about how to run our code with your dataset.

* Training 
  * Input: 
    * raw text file, one sentence per line. For example,
    ```
    a bicycle replica with a clock as the front wheel
    a black honda motorcycle parked in front of a garage
    ```
  * Output: 
    * the program will save the trained model automatically
    * a generation file predicted by generator, namely `<EPOCH>_examplar_gen`, e.g., `054_examplar_gen`
    ```angular2html
    a car is sitting on a red fire hydrant
    two people are standing on a train
    a bathroom with a toilet and a sink
    ```
    * reconstruction file predicted by autoencoder, namely `autoencoder.txt`  
    * print out `epoch`, `learning rate`, `loss`, `perplexity`, `accuracy` like below:
    ```angular2html
    | epoch   6 |     0/  453 batches | lr 0.000000 | ms/batch  1.36 | loss  0.02 | ppl     1.02 | acc     0.38 | train_ae_norm     1.00
    [6/100][199/453] Loss_D: 1.33532667 (Loss_D_real: 0.66654706 Loss_D_fake: 0.66877961) Loss_G: 0.00273732 Loss_Enh_Dec: -0.00751842
    | epoch   6 |   200/  453 batches | lr 0.000000 | ms/batch 765.74 | loss  3.66 | ppl    38.99 | acc     0.39 | train_ae_norm     1.00
    [6/100][399/453] Loss_D: 1.34199166 (Loss_D_real: 0.67177409 Loss_D_fake: 0.67021751) Loss_G: -0.00233009 Loss_Enh_Dec: -0.00531169
    | epoch   6 |   400/  453 batches | lr 0.000000 | ms/batch 777.91 | loss  3.67 | ppl    39.16 | acc     0.39 | train_ae_norm     1.00
    | end of epoch   6 | time: 359.45s | test loss  3.65 | test ppl 38.61 | acc 0.393
    ```
* Inference:
  * Step 1: train your own model and obtain `model.pt` and other related files under `results/coco_result2021-xx-xx-xx-xx-xx` (example for coco dataset)
  * Step 2: run `inference.py` with same model arguments(add noise/gan_d_local/...) as your training.
  * Input: generate noise by `fixed_noise = Variable(torch.ones(args.eval_batch_size, args.z_size).normal_(0, 1).to(device))`
  * Call the function `gen_fixed_noise(fixed_noise, gen_text_savepath)` defined in `train.py`
  * Output: a generation file predicted by generator, namely `inference_result` under your model dir `results/coco_result2021-xx-xx-xx-xx-xx` (example for coco dataset)
  * Example Command for coco dataset
  ```python
  # Train
  python train.py --data_path data/MS_COCO --maxlen 15 --save ./results/coco_result \
  --batch_size 256 --niters_gan_d 1 --niters_gan_ae 1 --lr_gan_g 4e-04 --lr_ae 0.12 \
  --add_noise --gan_d_local --enhance_dec
  # Infer
  python inference.py --save results/coco_result2021-xx-xx-xx-xx --data_path data/MS_COCO --add_noise --gan_d_local --enhance_dec
  ```
  * Example Output
  ```angular2html
  a car is sitting on a red fire hydrant
  two people are standing on a train
  a bathroom with a toilet and a sink
  ```
    
## Acknowledgement
Thanks to the source code provider [ARAE](https://openreview.net/forum?id=BkM3ibZRW)
