from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer 
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation

from itertools import chain, combinations
from data import cfg, set_cfg, set_dataset
import threading, queue
import numpy as np
import torch, gc
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time,os,random,cProfile,pickle,json

from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image
from imutils.video import WebcamVideoStream

import matplotlib.pyplot as plt
import cv2
from multiprocessing.pool import ThreadPool
from queue import Queue

import pycocotools
import scipy.spatial.distance as dist
from GPUtil import showUtilization as gpu_usage

torch.multiprocessing.set_start_method('spawn', force=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##  Setting up torch for gpu utilization
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


cfg.mask_proto_debug = False
iou_thresholds = [x / 100 for x in range(80, 100, 5)]       ## Change this value in range of 40-90 for designated performances
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

## Creating dictinary to store logs 
log = {"total_person": 0,"total_person_in_red_zone": 0 , "total_person_in_green_zone": 0}


class SocialDistance:
    def __init__(self,id):
        # self.cap = cv2.VideoCapture(id)
        self.cap = WebcamVideoStream(src = id).start()
        self.width = 1280 #640#
        self.height = 720 #360#
        self.display_lincomb = False
        self.crop = True
        self.score_threshold = 0.15
        self.top_k = 30
        self.display_masks = True
        self.display_fps = False
        self.display_text  = True
        self.display_bboxes = True
        self.display_scores = False
        
        self.fast_nms = True
        self.cross_class_nms =True
        self.config = 'yolact_plus_base_config'
        print('Config specified. Parsed %s from the file name.\n' % self.config)
        set_cfg(self.config)
        print('Loading model...', end='')
        self.trained_model = 'weights/yolact_plus_base_54_800000.pth'
        self.model = Yolact()
        self.model.load_weights(self.trained_model)
        self.model.detect.use_fast_nms = self.fast_nms
        self.model.detect.use_cross_class_nms = self.cross_class_nms
        self.model.eval() 
        self.model = self.model.to(device,non_blocking=True)
        print(' Done.')
        self.model_path = SavePath.from_str(self.trained_model)
        
    
    def prep_display(self,dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        
        lineThickness = 2
        

        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape
        
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(dets_out, w, h, visualize_lincomb = self.display_lincomb,
                                            crop_masks        = self.crop,
                                            score_threshold   = self.score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            # idx = t[1].argsort(0, descending=True)[top_k]
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][:self.top_k]

            classes, scores, boxes = [x[:self.top_k].cpu().detach().numpy() for x in t[:3]]
        
        num_dets_to_consider = min(self.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < self.score_threshold:
                num_dets_to_consider = j
                break

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
            
            if on_gpu is not None and color_idx in color_cache[on_gpu]:
                return color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    color_cache[on_gpu][color_idx] = color
                return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if self.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]
            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1
            
            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
            
        if self.display_fps:
                # Draw the box for the fps on the GPU
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

            img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().detach().numpy()

        if self.display_fps:
            # Draw the text on the CPU
            text_pt = (4, text_h + 2)
            text_color = [255, 255, 255]

            cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        if num_dets_to_consider == 0:
            return img_numpy

        if self.display_text or self.display_bboxes:
            distance_boxes = []
            
            
            def all_subsets(ss):
                return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))
            

            def draw_distance(boxes):
                """
                    input : boxes(type=list)
                    Make all possible combinations between the detected boxes of persons
                    perform distance measurement between the boxes to measure distancing
                
                """
                red_counter = 0                        ## Countting people who are in high risk
                green_counter = 0
                for subset in all_subsets(boxes):
                    if len(subset)==2:
                        a = np.array((subset[0][2], subset[0][3]))
                        b = np.array((subset[1][2], subset[1][3]))
                        dist = np.linalg.norm(a-b)      ## Eucledian distance if you want differnt ways to measure distance b/w two boxes you can use the following options
                        # dist = spatial.distance.cosine(a, b)
                        # # print ('Eucledian distance is version-1', dist)
                        # # print ('Eucledian distance is', spatial.distance.euclidean(a, b))
                        # print ('Cosine distance is', dist)
                        if dist < 250 :
                            red_counter += len(subset)
                            cv2.line(img_numpy, (subset[0][2], subset[0][3]), (subset[1][2], subset[1][3]), (0,0,255) , lineThickness)
                            
                        elif dist < 300:
                            green_counter += len(subset)
                            cv2.line(img_numpy, (subset[0][2], subset[0][3]), (subset[1][2], subset[1][3]), (0,255,0) , lineThickness)
                    log["total_person_in_red_zone"] = red_counter//2
                    log["total_person_in_green_zone"] = green_counter//2
                    # gc.collect()
                    
            
            
            
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]

                if self.display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                if self.display_text:
                    _class = cfg.dataset.class_names[classes[j]]
                    if _class == "person":
                        log["total_person"] = num_dets_to_consider
                        distance_boxes.append(boxes[j, :].tolist())
                        draw_distance(distance_boxes)

                    text_str = '%s: %.2f' % (_class, score) if self.display_scores else _class

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        
        return img_numpy
    
    
    def main(self):
        q = queue.Queue()
        while True:
            def frame_render(queue_from_cam):
                frame = self.cap.read() # If you capture stream using opencv (cv2.VideoCapture()) the use the following line
                # ret, frame = self.cap.read()
                frame = cv2.resize(frame,(self.width, self.height))
                queue_from_cam.put(frame)
            cam = threading.Thread(target=frame_render, args=(q,))
            cam.start()
            cam.join()
            inputs = q.get()
            q.task_done()
            
            ## Desiging the frame with necessary infos
            title = "Social Distance Monitoring - COVID19"
            total_person = "Total = {}".format(log["total_person"])
            # print(log)
            red_zone = "High Risk = {}".format(log["total_person_in_red_zone"])
            green_zone = "Safe Distance = {}".format(log["total_person_in_green_zone"])
            notification_bar_thickness = 3

            overlay = inputs.copy()
            background = inputs.copy()
            opacity = 0.4
            
            cv2.rectangle(overlay, (0, 0), (1280, 100), (255,255,255), -1)
            cv2.rectangle(overlay, (0, 615), (400, 720), (255,255,255), -1)
            cv2.addWeighted(overlay,opacity,background,1-opacity,0, inputs)

            cv2.putText(inputs,title, (195,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)               ### Text Main Title
            cv2.putText(inputs,total_person, (50,640), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)      ### Text Total Person

            cv2.line(inputs, (15,660), (40,660), (0,0,255) , notification_bar_thickness)                             ### Line red-zone
            cv2.putText(inputs,red_zone, (50,670), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)        ### Text Red Zone Person

            cv2.line(inputs, (15,700), (40,700), (0,255,0) , notification_bar_thickness)                             ### Line Green-zone
            cv2.putText(inputs,green_zone, (50,710), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)      ### Text green Zone Person
            
            with torch.no_grad():
                inputs = torch.from_numpy(inputs).cuda().float()
                images = FastBaseTransform()(inputs.unsqueeze(0))
                images = images.to(device )
                preds = self.model(images)
                frame = self.prep_display(preds, inputs, None, None, undo_transform=False)

            
            ret, jpeg = cv2.imencode('.jpg', frame)
            torch.cuda.empty_cache()
            return jpeg.tostring()
            
            
            
            

