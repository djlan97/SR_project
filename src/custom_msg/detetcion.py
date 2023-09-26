#!usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models
import rospy

from skimage.filters import gaussian
#import os
from PIL import Image
import torchvision.transforms as T
#from pycocotools.coco import COCO
from dataset_processing.grasp import detect_grasps
from dataset_processing import image
from operator import itemgetter
from custom_msg.msg import custom
from grasp_model import MPA_SegmentationNetwork

categories = {0: {'id': 0, 'name': 'fruit', 'supercategory': 'none'},
              1: {'id': 1, 'name': 'apple', 'supercategory': 'fruit'},
              2: {'id': 2, 'name': 'banana', 'supercategory': 'fruit'},
              3: {'id': 3, 'name': 'can', 'supercategory': 'fruit'},
              4: {'id': 4, 'name': 'screwdriver', 'supercategory': 'fruit'}}

def get_rgb_file(filepath, normalise=True):
    rgb_img = image.Image.from_file(filepath)

    if normalise:
        rgb_img.normalise()
        rgb_img.img = rgb_img.img.transpose((2, 0, 1))
    return rgb_img.img

def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
    :param q_img: Q output of GG-CNN (as torch Tensors)
    :param cos_img: cos output of GG-CNN
    :param sin_img: sin output of GG-CNN
    :param width_img: Width output of GG-CNN
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.data.detach().cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).data.detach().cpu().numpy().squeeze()
    width_img = width_img.data.detach().cpu().numpy().squeeze() * 150.0

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img



def grasp_info_one(filepath):
    output=[]
    # model = torch.load('Weights/grasp_modelV2.pth', map_location=torch.device('cpu'))
    model = MPA_SegmentationNetwork(layers=[3,3,1,1],embed_dims=[64,128,256,512],mlp_ratios=[4,4,4,4],num_classes=1)
    model.load_state_dict(torch.load('Weights/grasp_modelV2_state_dict.pth', map_location=torch.device('cpu')))
    #model_classification = torchvision.models.resnet50(pretrained=True)



    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rgb_img = image.Image.from_file(filepath)
    # rgb_img = imread('dataset/08/pcd0856r.png')
    '''
    input = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    input = input.type(torch.float)
    input = input.to(device)
    pos_pred, cos_pred, sin_pred, width_pred = model(input)
    q_out, ang_out, w_out = post_process_output(pos_pred, cos_pred,sin_pred,width_pred)
    gs_1 = detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=1)
    gs_1
    '''
    rgb_img.normalise()
    rgb_img.img = rgb_img.img.transpose((2, 0, 1))

    x = torch.from_numpy(np.expand_dims(rgb_img.img, 0).astype(np.float32))
    pos_pred, cos_pred, sin_pred, width_pred = model(x)
    q_out, ang_out, w_out = post_process_output(pos_pred, cos_pred, sin_pred, width_pred)
    gs_1 = detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=1)
    g_center = gs_1[0].center
    g_angle = gs_1[0].angle
    g_width = gs_1[0].width
    g_cy = g_center[0]
    g_cx = g_center[1]

    input = Image.open(filepath)
    transform = T.Resize((224,224))
    transform1 = T.ToTensor()
    input = transform(input)
    input = transform1(input)
    input = input.unsqueeze(dim=0)
    #input = data_transforms['train'](rgb_img.img)
    #output_class = model_classification(input)
    #_, preds = torch.max(output_class, 1)

    #print(preds)

    output = [g_cx,g_cy,g_angle,g_width]

    fig = plt.figure(figsize=(20, 10))

    immage = (get_rgb_file(filepath, normalise=False))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(immage)

    ax.axis('off')
    for g in gs_1:
        #print(g.angle)
        g.plot(ax)
    #plt.show(block=False)
    plt.show()


    return output

def single_grasp(filepath):
    imge = Image.open(filepath)
    imge = imge.crop((0,0,640,320))
    imge.save('Ros_Gazebo_image/mod_framesingle.jpg')
    #cv2.imwrite('Ros_Gazebo_image/mod_frame0000.jpg',imge)
    # imge.show()

    output = grasp_info_one('Ros_Gazebo_image/mod_framesingle.jpg')
    return output

def multiple_grasps(filepath,n_graps):
    imge = Image.open(filepath)
    imge = imge.crop((0,0,640,320))
    imge.save('Ros_Gazebo_image/mod_frame00015test.jpg')
    output = grasp_info_multiple('Ros_Gazebo_image/mod_frame00015test.jpg',n_graps=n_graps)
    return output


def grasp_info_multiple(filepath,n_graps=4):
    # model = torch.load('Weights/grasp_modelV2.pth', map_location=torch.device('cpu'))
    model = MPA_SegmentationNetwork(layers=[3,3,1,1],embed_dims=[64,128,256,512],mlp_ratios=[4,4,4,4],num_classes=1)
    model.load_state_dict(torch.load('Weights/grasp_modelV2_state_dict.pth', map_location=torch.device('cpu')))
    outputs=[]
    output=[]
    rgb_img = image.Image.from_file(filepath)
    # rgb_img = imread('dataset/08/pcd0856r.png')
    '''
    input = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    input = input.type(torch.float)
    input = input.to(device)
    pos_pred, cos_pred, sin_pred, width_pred = model(input)
    q_out, ang_out, w_out = post_process_output(pos_pred, cos_pred,sin_pred,width_pred)
    gs_1 = detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=1)
    gs_1
    '''
    rgb_img.normalise()
    rgb_img.img = rgb_img.img.transpose((2, 0, 1))

    x = torch.from_numpy(np.expand_dims(rgb_img.img, 0).astype(np.float32))
    pos_pred, cos_pred, sin_pred, width_pred = model(x)
    q_out, ang_out, w_out = post_process_output(pos_pred, cos_pred, sin_pred, width_pred)
    gs_1 = detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=n_graps)
    for g in gs_1:
        g_center = g.center
        g_angle = g.angle
        g_width = g.width
        g_cy = g_center[0]
        g_cx = g_center[1]
        output = [g_cx,g_cy,g_angle,g_width]

        outputs.append((output))
        g_x = g.center[1]
        g_y = g.center[0]


    fig = plt.figure(figsize=(20, 10))
    immage = (get_rgb_file(filepath, normalise=False))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(immage)
    ax.axis('off')
    for g in gs_1:
        #print(g.angle)
        g.plot(ax)
    # plt.show(block=False)
    plt.show()

    return outputs

def img_ac_detection(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t = T.ToTensor()
    image = t(image)
    return image

def obj_detection(filepath):
    model = torch.load('Weights/detection_model.pth', map_location='cpu')
    im = img_ac_detection(filepath)

    model.eval()
    prediction = model([im])
    pred = prediction[0]
    final_pred = {}
    final_pred['boxes'] = pred['boxes'][pred['scores'] > 0.75]
    final_pred['labels'] = pred['labels'][pred['scores'] > 0.75]
    final_pred['scores'] = pred['scores'][pred['scores'] > 0.75]

    classes = [i[1]['name'] for i in categories.items()]
    img_int = torch.tensor(im * 255, dtype=torch.uint8)
    fig = plt.figure(figsize=(14, 10))
    plt.imshow(torchvision.utils.draw_bounding_boxes(img_int,
                                                     pred['boxes'][pred['scores'] > 0.75],
                                                     [classes[i] for i in
                                                      pred['labels'][pred['scores'] > 0.75].tolist()], width=4
                                                     ).permute(1, 2, 0))

    return final_pred

def solve(tl, br, p) :
   if (p[0] > tl[0] and p[0] < br[0] and p[1] < br[1] and p[1] > tl[1]) :
      return True
   else :
      return False

def compute_final_predictions(final_pred,grasp_output):
    det = []
    dets = []
    for i in range(len(final_pred['boxes'])):
        x1 = final_pred['boxes'][i, 0].item()
        y1 = final_pred['boxes'][i, 1].item()
        x2 = final_pred['boxes'][i, 2].item()
        y2 = final_pred['boxes'][i, 3].item()
        c = final_pred['labels'][i].item()
        det = [x1, y1, x2, y2, c]
        dets.append(det)
    sorted_dets = sorted(dets, key=itemgetter(2), reverse=True)
    final_prediction = []
    final_predictions = []
    for d in sorted_dets:
        bl = [d[0], d[1]]
        tr = [d[2], d[3]]
        for g in grasp_output:
            p = [g[0], g[1],g[2]]
            if solve(bl, tr, p):
                final_prediction = [p[0], p[1],p[2], d[4]]
                final_predictions.append(final_prediction)
                break
    return final_predictions

if __name__ == "__main__":
    msg = custom()
    msg.x1 = -1.0
    msg.y1 =  -1.0
    msg.a1 = -1.0
    msg.c1 = -1
    msg.x2 = -1.0
    msg.y2 =  -1.0
    msg.a2 = -1.0
    msg.c2 = -1
    msg.x3 = -1.0
    msg.y3 =  -1.0
    msg.a3 = -1.0
    msg.c3 = -1
    msg.x4 = -1.0
    msg.y4 =  -1.0
    msg.a4 = -1.0
    msg.c4 = -1
    #filepath è l'indirizzo dove hai memorizzato l'immagine presa da gazebo
    filepath = 'world_images/my_world_04.jpg'
    final_pred = obj_detection(filepath)
    print(len(final_pred['boxes']))
    if len(final_pred['boxes']) == 1:
        output = single_grasp(filepath)
        output.append(final_pred['labels'].item())
        msg.x1=output[0]
        msg.y1 = output[1]
        msg.a1 = output[2]
        msg.c1 = output[4]
        print(msg)
    else:
        output=multiple_grasps(filepath,n_graps=len(final_pred['boxes']))
        final_predictions = compute_final_predictions(final_pred=final_pred,grasp_output=output)
        #for i in range(len(final_predictions)):
        for i in range(4):
            if i == len(final_predictions):
                break
            if i == 0:
                msg.x1 = final_predictions[i][0]
                msg.y1 = final_predictions[i][1]
                msg.a1 = final_predictions[i][2]
                msg.c1 = final_predictions[i][3]
            if i == 1:
                msg.x2 = final_predictions[i][0]
                msg.y2 = final_predictions[i][1]
                msg.a2 = final_predictions[i][2]
                msg.c2 = final_predictions[i][3]
            if i == 2:
                msg.x3 = final_predictions[i][0]
                msg.y3 = final_predictions[i][1]
                msg.a3 = final_predictions[i][2]
                msg.c3 = final_predictions[i][3]
            if i == 3:
                msg.x4 = final_predictions[i][0]
                msg.y4 = final_predictions[i][1]
                msg.a4 = final_predictions[i][2]
                msg.c4 = final_predictions[i][3]

    rospy.init_node('det_pub')
    rospy.loginfo('Node has been started')

    pub = rospy.Publisher('/detection_topic',custom,queue_size=10)

    rate = rospy.Rate(1) # 2 times x s

    while not rospy.is_shutdown():

        pub.publish(msg)

        rate.sleep() #regola il loop secondo rate
        
        print(msg)
