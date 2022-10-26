
# Included bug fixed of the ALCNEet
Fix issued reported in the orginal scripts.

More features will come later after further testing. Including improved logging and file management, draw bounding box of the grond truth target in prediction images using the .xml files in the dataset.


***
# Description from the orginal author

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
