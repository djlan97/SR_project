import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from dataset_processing.grasp import detect_grasps,GraspRectangles
from dataset_processing import grasp, image
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from Image_Classification_Model import *
from PIL import Image
import torchvision.transforms as T

classes = ['apple', 'banana', 'can', 'spoon']

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
    model = torch.load('Weights/grasp_model.pth', map_location=torch.device('cpu'))
    model_classification = torch.load('Weights/model_classification.pth', map_location=torch.device('cpu'))



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
    output_class = model_classification(input)
    _, preds = torch.max(output_class, 1)
    print(output_class)
    print(f'class: {classes[preds]}')
    output = [g_cy,g_cx,g_angle,g_width,classes[preds]]

    fig = plt.figure(figsize=(20, 10))
    plt.title(classes[preds])
    immage = (get_rgb_file(filepath, normalise=False))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(immage)

    ax.axis('off')
    for g in gs_1:
        print(g.angle)
        g.plot(ax)
    plt.show(block=False)


    return output

def grasp_info_multiple(results,filepath,boxes,classes,n_graps=2):
    model = torch.load('Weights/grasp_model.pth', map_location=torch.device('cpu'))
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
        g_x = g.center[1]
        g_y = g.center[0]
        for i in results.flatten():
            x, y = boxes[i][0], boxes[i][1]
            width, height = boxes[i][2], boxes[i][3]
            if (abs(g_x - x) < abs(width/2)) and (abs(g_y - y) < abs(height/2)):
                x = [g_y,g_x,g.angle,g.width,int(classes[i])]
                outputs.append(x)

    fig = plt.figure(figsize=(20, 10))
    immage = (get_rgb_file(filepath, normalise=False))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(immage)
    ax.axis('off')
    for g in gs_1:
        print(g.angle)
        g.plot(ax)
    plt.show(block=False)

    return outputs

if __name__ == "__main__":

    output = grasp_info_one('Ros_Gazebo_image/frame0000.jpg')
    '''
    input = Image.open('Ros_Gazebo_image/frame0008.jpg')
    transform = T.Resize((224,224))
    transform1 = T.ToTensor()
    input = transform(input)
    input = transform1(input)
    print(input.shape)
    '''














