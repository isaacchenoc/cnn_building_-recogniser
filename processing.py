import os
import cv2
import numpy as np

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

test = []

'''
Uncomment the code section if you want to achieve different data preprocessing
'''
for file in os.listdir('png-ZuBuD'):
    if file != ".DS_Store" and file!='train' and file!='test':
        img = cv2.imread('png-ZuBuD/'+file)
        # uncomment if you want grayscale image
        #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY,1)
        img = cv2.resize(img,(480,480)) # image size
        # crop the image into 4
        '''
        im1 = img[0:240, 0:240]
        im2 = img[0:240, 240:480]
        im3 = img[240:480, 0:240]
        im4 = img[240:480, 240:480]
        '''
        # rotate the image 90, 180, 270 degrees
        '''
        h=img.shape[0]
        w=img.shape[1]
        m90 = cv2.getRotationMatrix2D((h/2,w/2), 90, 1.0)
        m180 = cv2.getRotationMatrix2D((h/2,w/2), 180, 1.0)
        m270 = cv2.getRotationMatrix2D((h/2,w/2), 270, 1.0)
        img_90 = cv2.warpAffine(img, m90, (w, h))
        img_180 = cv2.warpAffine(img, m180, (w, h))
        img_270 = cv2.warpAffine(img, m270, (w, h))
        '''
        # get average image
        #img_avg = 0.25*img+0.25*img_90+0.25*img_180+0.25*img_270
        if file[0:10] in test:
        #if file[16] != '5':
                createFolder('png-ZuBuD/train/'+file[0:10])
                imname = 'png-ZuBuD/train/'+file[0:10]+'/'
                imname1 = imname+'1'+file
                imname2 = imname+'2'+file
                imname3 = imname+'3'+file
                imname = imname+file 
                cv2.imwrite(imname,img)
                #cv2.imwrite(imname,img_avg) # average image
                '''
                # rotated image
                cv2.imwrite(imname1,img_90)
                cv2.imwrite(imname2,img_180)
                cv2.imwrite(imname3,img_270)
                cv2.imwrite(imname,img)                
                '''
                '''
                # crop image
                cv2.imwrite(imname1,im1)
                cv2.imwrite(imname2,im2)
                cv2.imwrite(imname3,im3)
                cv2.imwrite(imname,im4)
                '''
        else:        
                createFolder('png-ZuBuD/test/'+file[0:10])
                imname = 'png-ZuBuD/test/'+file[0:10]+'/'
                imname1 = imname+'1'+file
                imname2 = imname+'2'+file
                imname3 = imname+'3'+file
                imname = imname+file 
                cv2.imwrite(imname,img)
                #cv2.imwrite(imname,img_avg) # average image
                '''
                # rotated image
                cv2.imwrite(imname1,img_90)
                cv2.imwrite(imname2,img_180)
                cv2.imwrite(imname3,img_270)
                cv2.imwrite(imname,img)                
                '''
                '''
                # crop image
                cv2.imwrite(imname1,im1)
                cv2.imwrite(imname2,im2)
                cv2.imwrite(imname3,im3)
                cv2.imwrite(imname,im4)
                '''
                test.append(file[0:10])
        
