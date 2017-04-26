# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## MCWPrune

MCWPrune/Caffe is developed by MulticoreWare for squeezing Convolution and InnerProduct layers by pruning the network. This repository is an enhancement of [Dynamic Network Surgery](https://github.com/yiwenguo/Dynamic-Network-Surgery) with additional features and techniques to improve accuracy.

This fork of Caffe has two new additional layers namely SqueezeConvolution and SqueezeInnerProduct for pruning and splicing the Convolution and InnerProduct layers respectively. To enable these layers, use layer type as  SqueezeConvolution and SqueezeInnerProduct in the prototxt and for the layer params - squeeze_convolution_param and squeeze_inner_product_param refer the example below: 
  ~~~~
  squeeze_convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
    gamma: 0.00001
    power: 1
    c_rate: 0.3
    iter_stop: 40000
    weight_mask_filler {
      type: "constant"
      value: 1
    }
    bias_mask_filler {
      type: "constant"
      value: 1
    }
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
~~~~

C_rate controls the rate for pruning for every layer. Higher the c_rate, more the pruning happens.
For improving accuracy of the pruned caffemodel, techniques like Retraining and Dynamic Splicing are implemented in the code.

 For retraining and Dynamic Splicing, turn on the corresponding flags in squeeze_conv_layer.h and squeeze_inner_product_layer.h. 
 
Other tools for processing the caffemodel - applying the masks on weights, printing the compression factor, truncating masks from the caffemodel can be found in miscellaneous_scripts folder.

The code changes are made upon Caffe-rc5 tag.


## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
