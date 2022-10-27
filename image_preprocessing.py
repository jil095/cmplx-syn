"""Preprocess the VGGFace2 images with SEnet50"""
from __future__ import print_function
from __future__ import division
import csv,os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch
from torch.autograd import Variable
import imp

def resize_crop(image,desired_size = 256,crop_size=224):
    
    old_size = np.array(image.shape[:2])
    ratio = float(desired_size)/min(old_size)
    new_size = (old_size*ratio).astype(np.int32)
    delta_h =  (new_size[0]-crop_size)//2
    delta_w = (new_size[1]-crop_size)//2
    im=Image.fromarray(image)
    im = im.resize(new_size[::-1], Image.ANTIALIAS)
    im=np.array(im)
    im=im[delta_h:delta_h+crop_size,delta_w:delta_w+crop_size]
    return im

MainModel = imp.load_source('MainModel', 'senet50_ft_pytorch/senet50_ft_pytorch.py') 
model = torch.load('senet50_ft_pytorch/senet50_ft_pytorch.pth')
model.eval()
desired_size=256
crop_size=224
name_list=[]
face_list=[]
with open('bb_landmark/loose_bb_train.csv') as csv_file:
    csv_reader = list(csv.reader(csv_file, delimiter=','))
current_name=None
for row in csv_reader[1:]:
    NAME_ID,X,Y,W,H=row
    if current_name is None:
        current_name=NAME_ID[:7]
        current_num=0
    if current_name!=NAME_ID[:7]:
        print(current_name,current_num)
        with open("name_and_pattern/%s.pkl" % current_name,"wb") as f:
            pickle.dump([name_list,face_list],f, protocol=2)
        current_name=NAME_ID[:7]
        current_num=0
        
    X,Y,W,H=int(X),int(Y),int(W),int(H)
    X1,X2=X-int(W/2*0.3),X+W+int(W/2*0.3)
    Y1,Y2=Y-int(H/2*0.3),Y+H+int(H/2*0.3)
    img=mpimg.imread(os.path.join('vggface2_train',NAME_ID+'.jpg'))
    if X1<0 or Y1<0 or X2>img.shape[1] or Y2>img.shape[0]:
        continue
    # plt.subplot(1,4,1)
    # imgplot = plt.imshow(img)
    # plt.subplot(1,4,2)
    # imgplot = plt.imshow(img[:,X:X+W][Y:Y+H])
    # plt.subplot(1,4,3)
    # imgplot = plt.imshow(img[:,X1:X2][Y1:Y2])
    image=img[:,X1:X2][Y1:Y2]
    image = np.array(resize_crop(image))
    # plt.subplot(1,4,4)
    # imgplot = plt.imshow(image)
    image_bgr=image[:,:,::-1]-np.reshape([91.4953, 103.8827, 131.0912],[1,1,3]) 
    image_bgr=np.expand_dims(np.transpose(image_bgr,(2,0,1)),0)
    #--> N,C_in,H,W
    model.alsolastlayer=True
    if model.alsolastlayer:
        outputs = model(Variable(torch.FloatTensor(image_bgr)))
        
        name_list.append(NAME_ID)
        face_list.append(outputs.data.numpy().reshape([2048]))
    else:
        outputs = model(Variable(torch.FloatTensor(image_bgr)))
        _, predicted = torch.max(outputs.data, 1)
        print(NAME_ID,predicted[0][0])
        
    current_num+=1
print(current_name,current_num)

with open("name_and_pattern/%s.pkl" % current_name,"wb") as f:
    pickle.dump([name_list,face_list],f, protocol=2)