import os, sys
import cv2
import time
import numpy as np
import mxnet as mx
from mxnet import nd
import gluoncv as gcv
from mxnet.gluon.nn import BatchNorm
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../MobileFace_Detection/utils/')
from data_presets import data_trans
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../MobileFace_Detection/')
from mobilefacedetnet import mobilefacedetnet_v2
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../MobileFace_Tracking/')
from mobileface_sort_v1 import Sort

class face_inference(object):
    def __init__(self):
        self.ctx = [mx.gpu(0)]
        self.ctx = [mx.cpu()] if not self.ctx else self.ctx

        self.net = mobilefacedetnet_v2('mobilefacedet_v2_gluoncv.params')
        self.net.set_nms(0.45, 200)
        self.net.collect_params().reset_ctx(ctx = self.ctx)

        self.mot_tracker = Sort(10, 3) 

        self.img_short = 256   
        self.colors = np.random.rand(32, 3) * 255

    def inference(self, frame):
        #cv2.imshow('img', frame)  
        img_short = 256 
        dets = []
        # ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_nd = nd.array(frame_rgb)
        x, img = data_trans(frame_nd, short=self.img_short)
        x = x.as_in_context(self.ctx[0])
        #ids, scores, bboxes = [xx[0].asnumpy() for xx in result]
        tic = time.time()
        result = self.net(x)
        toc = time.time() - tic
        print('Detection inference time:%fms' % (toc*1000))
        ids, scores, bboxes = [xx[0].asnumpy() for xx in result]
        return result
       

