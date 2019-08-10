# Residual Dense Network for Image Super-Resolution
TensorFlow implementatoin of Residual Dense Network for Image Super-Resolution, paper click [here](https://arxiv.org/abs/1802.08797)

References: [Author's version](https://github.com/yulunzhang/RDN), [PyTorch version](https://github.com/lingtengqiu/RDN-pytorch), [Another tensorflow](https://github.com/hengchuan/RDN-TensorFlow/blob/master/README.md)

I have to admmit, I didn't get the result as good as the author said in paper(about 1dB PSNR low), hope someone can get that.

## Requirements
make sure you have these lib below:
 - TensorFlow==1.8.0
 - NumPy==1.15.4
 - h5py==2.9.0
 - imageio==2.4.1
 - opencv-python==3.4.3.18
 - Pillow==6.1.0
 - scikit-image==0.14.1
 - scikit-learn==0.21.3
 - scipy==1.1.0

## Train
To train the model, put you training set in *./data*, for example, to train on DIV2K_train_HR, put folder *DIV2K_train_HR* in folder *data*.

For first training, type:
```shell
python train.py --dataset DIV2K_train_HR --image_form .png --fresh True --scale 2
```
For fine-tune, type:
```shell
python train.py --dataset DIV2K_train_HR --image_form .png --weight_file rdn.ckpt-100000
```
*--dataset* indicate which dataset you gonna use, *--image_for* indicate what type of images in this dataset, *--fresh* means it's trianing from scratch, **DO NOT** type *--fresh* if you are in fune-tune stage, and *--weight_file* indicate ckpt files you gonna restore and continue to train.

in Training, you will see:
```shell
Training step: [   100], time: [0.174093min], loss: [1.412353]
Training step: [   200], time: [0.256196min], loss: [0.987543]
...
```
then you on training.

When trainig, you can open **TensorBoard** to see the metrics, type:
```shell
cd ~/RDN-TensorFlow
tensorboard --logdir=cache/event
```
Then open the website showed on screen

## Test
After training, type:
```shell
python valid.py --dataset Set5 --image_form .png --weight_file rdn.ckpt-200000 --save true
```
and you will see the output...

---

Any questions, email: cokespace2@gmail.com