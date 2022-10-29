
**Master branch**

Fix reported issues in the orginal scripts.

**[Dev branch](https://github.com/Victor4869/open-alcnet/tree/dev)**

This branch contains additional features and improvements, check the description for this branch for more detail.

***
If you are unfamiliar with how to setup an environment for MXNet or how to run the scripts, you can find the tutorial in the [Wiki](https://github.com/Victor4869/open-alcnet/wiki).
***
# open-alcnet

codes and trained models for the ALCNet

## Requirements
 
Install [MXNet](https://mxnet.apache.org/) and [Gluon-CV](https://gluon-cv.mxnet.io/):

```
pip install --upgrade mxnet-cu100 gluoncv
```

## Dataset

The SIRST dataset: <https://github.com/YimianDai/sirst>

## Experiments 

The trained model params are in `./params`

## Citation

Please cite our paper in your publications if our work helps your research. BibTeX reference is as follows.

```
@inproceedings{dai21acm,
  title   =  {Asymmetric Contextual Modulation for Infrared Small Target Detection},
  author  =  {Yimian Dai and Yiquan Wu and Fei Zhou and Kobus Barnard},
  booktitle =  {{IEEE} Winter Conference on Applications of Computer Vision, {WACV} 2021}
  year    =  {2021}
}

@article{TGRS21ALCNet,
  author    = {{Dai}, Yimian and {Wu}, Yiquan and {Zhou}, Fei and {Barnard}, Kobus},
  title   = {{Attentional Local Contrast Networks for Infrared Small Target Detection}},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  pages   = {1--12},
  year    = {2021},
}
```
