import json
import pylab as pl
import random
import numpy as np
import cv2
import copy

type45="i2,i4,i5,il100,il60,il80,io,ip,p10,p11,p12,p19,p23,p26,p27,p3,p5,p6,pg,ph4,ph4.5,ph5,pl100,pl120,pl20,pl30,pl40,pl5,pl50,pl60,pl70,pl80,pm20,pm30,pm55,pn,pne,po,pr40,w13,w32,w55,w57,w59,wo"
type45 = type45.split(',')

def load_img(annos, datadir, imgid):
    img = annos["imgs"][imgid]
    imgpath = datadir+'/'+img['path']
    imgdata = pl.imread(imgpath)
    #imgdata = (imgdata.astype(np.float32)-imgdata.min()) / (imgdata.max() - imgdata.min())
    if imgdata.max() > 2:
        imgdata = imgdata/255.
    return imgdata

def load_mask(annos, datadir, imgid, imgdata):
    img = annos["imgs"][imgid]
    mask = np.zeros(imgdata.shape[:-1])
    mask_poly = np.zeros(imgdata.shape[:-1])
    mask_ellipse = np.zeros(imgdata.shape[:-1])
    for obj in img['objects']:
        box = obj['bbox']
        cv2.rectangle(mask, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
        if 'polygon' in obj and len(obj['polygon'])>0:
            pts = np.array(obj['polygon'])
            cv2.fillPoly(mask_poly, [pts.astype(np.int32)], 1)
            # print pts
        else:
            cv2.rectangle(mask_poly, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
        if 'ellipse' in obj:
            rbox = obj['ellipse']
            rbox = ((rbox[0][0], rbox[0][1]), (rbox[1][0], rbox[1][1]), rbox[2])
            print (rbox)
            cv2.ellipse(mask_ellipse, rbox, 1, -1)
        else:
            cv2.rectangle(mask_ellipse, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
    mask = np.multiply(np.multiply(mask,mask_poly),mask_ellipse)
    return mask
    
def draw_all(annos, datadir, imgid, imgdata, color=(0,1,0), have_mask=True, have_label=True):
    img = annos["imgs"][imgid]
    if have_mask:
        mask = load_mask(annos, datadir, imgid, imgdata)
        imgdata = imgdata.copy()
        imgdata[:,:,0] = np.clip(imgdata[:,:,0] + mask*0.7, 0, 1)
    for obj in img['objects']:
        box = obj['bbox']
        cv2.rectangle(imgdata, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), color, 3)
        ss = obj['category']
        if 'correct_catelog' in obj:
            ss = ss+'->'+obj['correct_catelog']
        if have_label:
            cv2.putText(imgdata, ss, (int(box['xmin']),int(box['ymin']-10)), 0, 1, color, 2)
    return imgdata


