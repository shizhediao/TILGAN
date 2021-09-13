# Conditional Generation Code

This is the implementaition for Conditional Generation Task.

## Quick Links
- [Conditional Generation Code](#conditional-generation-code)
  - [Quick Links](#quick-links)
  - [Environment](#environment)
  - [Quick Start](#quick-start)
  - [Use with your own data](#use-with-your-own-data)
    - [Training](#training)
    - [Inference](#inference)
  - [Hyper-parameter Setting](#hyper-parameter-setting)
  - [Acknowledgement](#acknowledgement)

## Environment 
To run our code, please install all the dependency packages by using the following command under python 3.6 (recommend):

```
pip install -r requirements.txt
```
Note that to use `tensorflow-gpu==1.13.1`, you need to install and use `CUDA 10.0`.

## Quick Start
After the environment setup, you can simply type the following command:

```shell
mkdir -p output/tilgan model
python train.py --gpu_device <GPU_ID>
where GPU_ID=0/1/2/... the GPU you want to use
```

## Use with your own data
Our code allows you to do conditional generation tasks, for example, story completion with your own data.
In this section, we provide instructions about how to run our code with your dataset.
### Training
* Input: Put your own data under `text_data` and rename them as `train.txt`, `valid.txt`, `test.txt`. The format is a five sentence story per line, e.g.,:
```angular2html
one day , i was driving the van down to the store . when i got to the stop sign , the check engine light started flashing . i panicked and carefully drove the van to the nearest mechanic shop . they checked it out but could not repair the van . the van had to be sold for parts and i had to get a new vehicle .
```
* Preprocess:
  Run `python preprocess.py`

You will get the processed ids of a story per line and the word2idx mapping used in this task is `./data/vocab_20000`. And each sentence is seperated by `-1`. For example,
```angular2html
32 34 13 16 8 310 5 2424 88 6 5 98 4 -1 35 16 37 6 5 312 901 13 5 539 1839 624 85 5333 4 -1 16 1983 11 1025 253 5 2424 6 5 2685 1705 418 4 -1 22 520 15 28 36 49 51 2100 5 2424 4 -1 5 2424 19 6 63 611 20 2040 11 16 19 6 58 7 45 2707 4 -1
```

* Output: 
  * the program will save the trained model `model/<your_trained_model>` automatically
  * print out `global step`, `loss`, `perplexity`, `discriminator loss`, `generator loss`, `generator_autoencoder loss` like below:
  ```angular2html
  global step 100   step-time 0.82s  loss 77.003 ppl 1125.74  disc 0.210 gen -0.183 gan_ae -0.102
  global step 200   step-time 0.63s  loss 65.980 ppl 398.93  disc 0.239 gen -0.163 gan_ae -0.141
  global step 300   step-time 0.63s  loss 65.455 ppl 392.00  disc 0.123 gen 0.005 gan_ae -0.109
  ```

### Inference
- Step 1: Prepare your data with format 4 sentences + one <pending_infer>  e.g. `S1. S2. <pending_infer>. S4. S5.` in one raw.
- Step 2: Put your trained model ckpt under `./<ckpt_path>` and your prepared data as `inference_data/infer.txt`
- Step 3: Preprocess your inference data same as previous training section, name it as infer.ids  within same folder as `infer.txt`
- Step 4: Run `mkdir inference` which is the output path.
- Step 5: Run python inference.py --model_dir `<ckpt_path>`
Example inference data are under `./inference_data`, please put your ckpt under `./inference_ckpt_example/tilgan`, follow the same sturcture (.index, .meta and .data) and simplely run "python inference.py --gpu_device 0" to have a try.
* Example Input
```angular2html
<pending_infer>. when i got to the stop sign , the check engine light started flashing . i panicked and carefully drove the van to the nearest mechanic shop . they checked it out but could not repair the van . the van had to be sold for parts and i had to get a new vehicle .
the man won a contest . he went to the station to collect . <pending_infer>. he did n't really like the band . he tried to sell them back to the radio employees . 
```
* Example Output
```angular2html
i was driving my new van when i saw a truck pull over .
he bought a radio .
```
## Hyper-parameter Setting
The detailed hyper-parameter setting is reported in our paper. Please check at Appendix. 

## Acknowledgement
Thanks to the source code provider [T-CVAE](https://www.ijcai.org/proceedings/2019/727)
