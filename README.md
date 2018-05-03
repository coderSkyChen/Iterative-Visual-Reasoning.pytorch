# Iterative-Visual-Reasoning.pytorch
Reimplementation for **Iterative Visual Reasoning Beyond Convolutions**(CVPR2018)，i've reimplemented it on **pytorch** according to [endernewton/iter-reason](https://github.com/endernewton/iter-reason)

## Note
- This is a reimplementation of the system described in the paper according to the author's codes: [endernewton/iter-reason
](https://github.com/endernewton/iter-reason).
- The author [endernewton](https://github.com/endernewton) has published the codes for spatial reasoning, so this codes only contain the baseline model and the spatial reasoning model. Global reasoning with knowledge graph has not been added.
- I've tried to reimplemente the project strictly according to the author's codes. The `crop_and_resize` function is build on top of the `roi_align` function in [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn). Also the weight initialization for all the modules is kept the same as the original codes(normal,xavier).
- The pretrained backbone models come from pytorch pretrained models, using caffe pretrained models may get better results.
- For now, the result of this reimplementation is lower than that reported in the paper by 2%~3%. If you are seeking to reproduce the results in the original paper, please use the [official code](https://github.com/endernewton/iter-reason) based on tensorflow.
- Feel free to contact me if you encounted any issues.
## Mainly Depencies
- Pytorch-0.3
- Tensorboard(this is optional)
- Cython 
- opencv-python

## Data preparation
Set up data, here we use [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/) as an example.
```
mkdir -p data/ADE
cd data/ADE
wget -v http://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip
tar -xzvf ADE20K_2016_07_26.zip
mv ADE20K_2016_07_26/* ./
rmdir ADE20K_2016_07_26
# then get the train/val/test split
wget -v http://xinleic.xyz/data/ADE_split.tar.gz
tar -xzvf ADE_split.tar.gz
rm -vf ADE_split.tar.gz
cd ../..
```

## Compilation (for computing roi crop_and_resize)
```
cd ./lib
sh make.sh
cd ..
```
The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version." If you encounterd any issues, please refer to [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).

## Scripts for train_val, test
Note that you need to set the argument `DATA_DIR` in opts.py according to the dataset.

For the baseline model:
```
# Train_val:
CUDA_VISIBLE_DEVICES=0 python trainval_net.py --net res50 --cuda True --train_id 0.1 --iters 320000 --lr_decay_step 280000 --lr 0.0005

# Test
CUDA_VISIBLE_DEVICES=0 python test.py --net res50 --cuda True --train_id 0.1 --model_name your-model-name.pth

```
For the spatial reasoning model:
```
#Train_val:
CUDA_VISIBLE_DEVICES=0 python trainval_memory_net.py --net memory_res50 --cuda True --train_id 1.1 --iters 320000 --lr_decay_step 280000 --lr 0.0005

#Test:
CUDA_VISIBLE_DEVICES=0 python test_memory.py --net res50 --cuda True --train_id 0.1 --model_name your-model-name.pth

```

## Benchmarking
|model|per-instance AP|per-instance AC|per-class AP|per-class AC|
|--------------|:-----:|:-----:|:-----:|:-----:|
|Res50-baseline|0.654|0.655|0.380|0.314|
|Res50-local (training..)|-|-|-|-|

## References
```
@inproceedings{chen18iterative,
    author = {Xinlei Chen and Li-Jia Li and Li Fei-Fei and Abhinav Gupta},
    title = {Iterative Visual Reasoning Beyond Convolutions},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    Year = {2018}
}
```

## Demo
<img src="https://github.com/coderSkyChen/Iterative-Visual-Reasoning.pytorch/raw/master/Images_for_readme/ADE_val_00000127.jpg" height  = "500" alt="3" align=left />
 <img src="https://github.com/coderSkyChen/Iterative-Visual-Reasoning.pytorch/raw/master/Images_for_readme/ADE_val_00000813.jpg" height  = "500" alt="5" align=left />




