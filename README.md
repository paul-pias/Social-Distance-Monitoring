# Social-Distance-Monitoring

## Introduction
This repository holds the implementation of monitoring social distancing implied for COVID-19 using  [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)) for object detection. 

## User Guideline
**System Requirement**
- For better performance you will cuda version 10 or 10.1
 - Python3

**Installation**
```
    $ pip install -r requirements.txt
 ```
 #### For the installation of torch using "pip" kindly follow the instructions from [Pytorch](https://pytorch.org/)

First, you need to clone the repository using gitbash (if gitbash is already installed) or you can download the zip file.
```
    $ git clone https://github.com/paul-pias/Social-Distance-Monitoring.git
```

If want to see your output in your browser execute the "server.py" script or else run "inference.py" to execute it locally.

If you want to run the inference on a ip camera need to use **WebcamVideoStream** with the following command. 
```
    "rtsp://assigned_name_of_the_camera:assigned_password@camer_ip/"
```
If you want to use YOLACT++, compile deformable convolutional layers (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)). Make sure you have the latest CUDA toolkit installed from [NVidia's Website](https://developer.nvidia.com/cuda-toolkit).
```
    cd external/DCNv2
    python setup.py build develop
```

In the official Yolact repository there are several pre-trained model available.
|    Image Size            |Model File (-m)                       |Config (-c)                   |
|----------------|-------------------------------|-----------------------------|
|550|[yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing)            |yolact_resnet50            |
|550          |[yolact_darknet53_54_800000.pth](https://drive.google.com/file/d/1dukLrTzZQEuhzitGkHaGjphlmRJOjVnP/view?usp=sharing)           |yolact_darknet53            |
|550          |[yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing)|yolact_base|
|700|[yolact_im700_54_800000.pth](https://drive.google.com/file/d/1lE4Lz5p25teiXV-6HdTiOJSnS7u7GBzg/view?usp=sharing)            |yolact_im700            |
|550         |[yolact_plus_resnet50_54_800000.pth](https://drive.google.com/file/d/1ZPu1YR2UzGHQD0o1rEqy-j5bmEm3lbyP/view?usp=sharing)            |yolact_plus_resnet50            |
|550          |[yolact_plus_base_54_800000.pth](https://drive.google.com/file/d/15id0Qq5eqRbkD-N3ZjDZXdCvRyIaHpFB/view?usp=sharing)|yolact_plus_base|


### Things to consider

Download the pre-trained weights and save in the folder **weights**, then from your terminal run the following command based on your preference.

```
    python inference.py -m=weights/yolact_base_54_800000.pth -c=yolact_base -i 0
```
Here 0 as id passed if you want to run the inference on webcam feed. If you don't parse any argument it will run with the default values. You can tweak the following values according to your preference. 


|      Input          |Value                        |HTML                         |
|----------------|-------------------------------|-----------------------------|
|width, height |1280 x 720 |              
|display_lincomb         |`False             
|crop          |True 	| For better segmentation use this flag as True
|score_threshold |0.15  | Higher the value better the performace          
|top_k         | 30     | At max how many objects will the model consider to detect in a given frame        
|display_masks          |`True` | Draw segmentation 
|display_fps |False                        
|display_text          |True
|display_bboxes         |`True             
|display_scores           |False
|fast_nms |True            
|cross_class_nms         | True             
|display_text          |True

###  The Purpose!
Social Distancing is a way of protecting yourself and others around you by knowing the facts and taking appropriate precautions. To prevent the spread of COVID-19 few guidelines were provided by World Health Organzation (WHO) and other public health agency. Maintaining at least 1 metre (3 feet) distance between yourself and anyone who is coughing or sneezing or for the time being everyone was one of them. 

 
To measure distance between two person eucledian distance was used in this work. **Euclidean distance** or **Euclidean metric** is the "ordinary" [straight-line](https://en.wikipedia.org/wiki/Straight_line "Straight line")  [distance](https://en.wikipedia.org/wiki/Distance "Distance") between two points in [Euclidean space](https://en.wikipedia.org/wiki/Euclidean_space "Euclidean space"). 

The **Euclidean distance** between two points **p** and **q** is the length of the [line segment](https://en.wikipedia.org/wiki/Line_segment "Line segment") connecting them ![\overline{\mathbf{p}\mathbf{q}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/6d397a90d8e00a9fbb6e7eb908cda31009fde6ee).(https://wikimedia.org/api/rest_v1/media/math/render/svg/6d397a90d8e00a9fbb6e7eb908cda31009fde6ee).
In the [Euclidean plane](https://en.wikipedia.org/wiki/Euclidean_plane "Euclidean plane"), if **p** = (_p_1, _p_2) and **q** = (_q_1, _q_2) then the distance is given by

![{\displaystyle d(\mathbf {p} ,\mathbf {q} )={\sqrt {(q_{1}-p_{1})^{2}+(q_{2}-p_{2})^{2}}}.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/4febdae84cbc320c19dd13eac5060a984fd438d8)

This formula was applied in the **draw_distance(boxes)** function where we got all the bounding boxes of person class in a given frame from the model where each bounding is a regression value consisting **(x,y,w,h)** . Where x and y represent 2 co-ordinates of the person and w & h represent widh and height correspondingly. All combinations of boxes were found to calculate the distance between them. 

