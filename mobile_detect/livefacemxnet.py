import os, sys
import cv2
import time
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../MobileFace_Tracking/')
from mobileface_sort_v1 import Sort
import server_inference1 as si

cap = cv2.VideoCapture(0)
faces = si.face_inference()
while cv2.waitKey(1) < 0:
    ret, frame = cap.read()
    result=faces.inference(frame)
    dets = []
    img_short = 256
    mot_tracker = Sort(10, 3)
    colors = np.random.rand(32, 3) * 255 
    h, w, c = frame.shape
    scale = float(img_short) / float(min(h, w))
    ids, scores, bboxes = [xx[0].asnumpy() for xx in result]
    for i, bbox in enumerate(bboxes):
        if scores[i]< 0.5:
            continue
        xmin, ymin, xmax, ymax = [int(x/scale) for x in bbox]
        result = [xmin, ymin, xmax, ymax, ids[i], scores[i]]
        result = [xmin, ymin, xmax, ymax, ids[i]]
        dets.append(result)

    dets = np.array(dets)
    tic = time.time()
    trackers = mot_tracker.update(dets)
    toc = time.time() - tic
    print('Tracking time:%fms' % (toc*1000))
    for d in trackers:
        color = (int(colors[int(d[4]) % 32, 0]), int(colors[int(d[4]) % 32,1]), int(colors[int(d[4]) % 32, 2]))
        cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), color, 3)  
    cv2.imshow("facedetect", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
