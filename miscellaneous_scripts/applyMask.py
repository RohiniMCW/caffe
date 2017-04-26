#!/usr/bin/env python
#************Purpose: To apply the masks on weights and generate new caffemodel*********************************
#************Usage:   python applyMask.py <path of deploy.prototxt> <path of target caffemodel>*****************
#************Authors: Rohini Priya <rohini@multicorewareinc.com>************************************************
#*******************  Zibiah Esme <zibiah@multicorewareinc.com> ************************************************

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import sys

prototxt   = sys.argv[1] #Relative or Absolute path to the deploy file
caffemodel = sys.argv[2] #Relative or Absolute path to the caffemodel
net_inputdata = {}

print_config = True

def get_inputs(net):
  net_inputdata['num'] = net.blobs['data'].data.shape[0]
  net_inputdata['channels'] = net.blobs['data'].data.shape[1]
  net_inputdata['height'] = net.blobs['data'].data.shape[2]
  net_inputdata['width'] = net.blobs['data'].data.shape[3]

if __name__ == '__main__':

    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    cnet= caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(), cnet)

    get_inputs(net)
    net.blobs['data'].reshape(net_inputdata['num'],
                            net_inputdata['channels'],
                            net_inputdata['height'],
                            net_inputdata['width'])
 
    vision_layers = ["Convolution", "InnerProduct", "SqueezeConvolution", "SqueezeInnerProduct"]
    for i, layer in enumerate(cnet.layer):
      layer_name = layer.name
      if layer.type in vision_layers:
          name = layer.name
          filters = weights = net.params[layer_name][0].data
          biases = net.params[layer_name][1].data
          filters_mask= net.params[layer_name][2].data
          biases_mask = net.params[layer_name][3].data
          print "Applying mask and processing weights for", name
          if layer.type == "InnerProduct" or layer.type == 'SqueezeInnerProduct':
              for ii in range(0, len(weights)):
                  for j in range(0, len(weights[ii])):
                      weights[ii][j] = filters_mask[ii][j] * weights[ii][j] #Multiply masks and weights
          else:
              for ii in range(0, len(weights)):
                  for j in range(0, len(weights[ii])):
                      for k in range(0, len(weights[ii][j])):
                          for l in range(0, len(weights[ii][j][k])):
                              weights[ii][j][k][l] = filters_mask[ii][j][k][l] * weights[ii][j][k][l] #Multiply masks and weights
          net.params[layer_name][0].data.flat = weights

net.save("MaskApplied_" + caffemodel.split('/')[-1]);
print "> Done!"
