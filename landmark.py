import os
import sys
import cv2
import torch
from torch.autograd import Variable
import torch as T
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
import argparse

from posemodel import pose_model

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def get_landmark(model, oriImg):
    param_, model_ = config_reader()

    imageToTest = Variable(T.transpose(T.transpose(T.unsqueeze(torch.from_numpy(oriImg).float(),0),2,3),1,2),volatile=True).cuda()

    multiplier = [x * model_['boxsize'] / oriImg.shape[0] for x in param_['scale_search']]

    heatmap_avg = torch.zeros((len(multiplier),19,oriImg.shape[0], oriImg.shape[1])).cuda()
    paf_avg = torch.zeros((len(multiplier),38,oriImg.shape[0], oriImg.shape[1])).cuda()
    for m in range(len(multiplier)):
        scale = multiplier[m]
        h = int(oriImg.shape[0]*scale)
        w = int(oriImg.shape[1]*scale)
        pad_h = 0 if (h%model_['stride']==0) else model_['stride'] - (h % model_['stride'])
        pad_w = 0 if (w%model_['stride']==0) else model_['stride'] - (w % model_['stride'])
        new_h = h+pad_h
        new_w = w+pad_w

        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
        imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5

        feed = Variable(T.from_numpy(imageToTest_padded)).cuda()
        output1,output2 = model(feed)
        heatmap = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output2)

        paf = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output1)

        heatmap_avg[m] = heatmap[0].data
        paf_avg[m] = paf[0].data

    heatmap_avg = T.transpose(T.transpose(T.squeeze(T.mean(heatmap_avg, 0)),0,1),1,2).cuda()
    paf_avg     = T.transpose(T.transpose(T.squeeze(T.mean(paf_avg, 0)),0,1),1,2).cuda()
    heatmap_avg=heatmap_avg.cpu().numpy()
    paf_avg    = paf_avg.cpu().numpy()

    all_peaks = []
    peak_counter = 0

    #maps =
    for part in range(18):
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))

        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse

        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    return all_peaks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('save_prefix', type=str)
    args = parser.parse_args()

    weight_name = './model/pose_model.pth'
    # oriImg = cv2.resize(oriImg,(64,128))
    model = pose_model()
    model.load_state_dict(torch.load(weight_name))
    model.cuda()
    model.float()
    model.eval()

    for im in sorted(os.listdir(args.data_path)):
        test_image = os.path.join(args.data_path,im)
        oriImg = cv2.imread(test_image)

        if not oriImg.size:
            print im+' is empty.'
            continue

        all_peaks = get_landmark(model,oriImg)

        # land = np.zeros((oriImg.shape[0],oriImg.shape[1]))
        # for i in range(len(all_peaks)):
        #     peak = all_peaks[i]
        #     if len(peak)==0:
        #         continue
        #     land[peak[0][1],peak[0][0]]=1

        landmark_txt = open(os.path.join(args.save_prefix,im.split('.')[0]+'.txt'),'w')
        for i in range(len(all_peaks)):
            peak = all_peaks[i]
            if len(peak)>0:
                print>>landmark_txt,str(peak[0][1])+' '+str(peak[0][0])
            else:
                assert ('-1')
                print>>landmark_txt,'-1 -1'

        print im+" create."

    # cv2.imwrite(os.path.join(args.save_prefix,os.path.basename(test_image)),land*255)
    # import pdb; pdb.set_trace()
