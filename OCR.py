#ROTATING THE WORD BY THE CALCULATED ANGLE
def slant(skel,degree,a2):
    t=list(skel.shape)
    pi=22/7
    m=math.tan((degree*pi)/180)
    f=0
    for i in range(0,t[0]):
        for j in range(0,t[1]):
            if skel[i][j]==255:
                ab1=i
                f=1
                break
        if f==1:
            break
    f=0
    for i in range(t[0]-1,-1,-1):
        for j in range(0,t[1]):
            if skel[i][j]==255:
                ab2=i
                f=1
                break
        if f==1:
            break
    diff=a2-ab1
    c=0
    for i in range(ab1,a2+1):
        shift=int(m*(diff-c))
        c+=1
        if c==diff+1:
            break
        for j in range(0,t[1]-1):
            if skel[i][j]==255:
                skel[i][j]=0
                if j-shift<=0:
                    skel[i][0]=150
                else:
                    skel[i][j-shift]=150
    diff=ab2-a2
    c=0
    for i in range(a2,ab2+1):
        shift=int(m*c)
        c+=1
        if c==diff:
            break
        for j in range(0,t[1]):
            if skel[i][j]==255:
                skel[i][j]=0
                if j+shift>=t[1]:
                    skel[i][t[1]-1]=150
                else:
                    skel[i][j+shift]=150
    return skel

#CONVERTING THE WORD INTO SINGLE PIXEL FORMAT
def skeletonize(img):
    img = img.copy() 
    skel = img.copy()
    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break
    return skel

#CALCULATING THE MAXIMUM SUM SUBARRAY OF SIZE 30 TO FIND THE CORE REGION OF THE ALPHABET
#THE ARRAY IS FORMED BY HORIZONTAL DENSITY HISTOGRAM OF THE IMAGE
def maxSum(arr, n, k):
    res = 0
    start=0
    for i in range(k):
        res += arr[i]
    curr_sum = res
    for i in range(k, n):
        curr_sum += arr[i] - arr[i-k]
        if res < curr_sum:
            res=curr_sum
            start=i-k+1
    return start

#ANGLE CALCULATION FOR SLANT REMOVAL
def angle(skel):
    l=[]
    t=skel.shape
    for i in skel:
        count=0
        for j in i:
            if j==255:
                count+=1
        l.append(count)
    flag=0
    if t[0]>=30:
        optimal_point=maxSum(l,t[0],30)
        ab1=optimal_point
        if optimal_point+30>=t[0]:
            optimal_point=t[0]-1
        else:
            optimal_point+=30
    else:
        #the no. of rows in the image are less than 30
        flag=1
        optimal_point=t[0]-1
        ab1=0   
    ab2=optimal_point    
    for i in range(0,t[1]):
        if skel[ab2][i]==255 or skel[ab2-1][i]==255:
            ab3=i
            break
    if flag==0:
        base=30
        pi=22/7
        max1=-9999
        if t[1]>ab3+45:
            c=45
        else:
            c=abs(t[1]-ab3)
        for i in range(0,c):
             m=math.atan2(i,base)
             degree = m*(180/pi)
             slope=i/base
             y1=ab3+i
             count=0
             for x_final in range(ab1,ab2+1):
                 y_final=int(slope*(ab1-x_final)+y1)
                 if skel[x_final][y_final]==255:
                     count+=1
             if max1<count:
                 max1=count
                 deg_final=degree
                 yy=y1
                 slope1=slope
        skel1=slant(skel,deg_final-5,ab2)
    else:
        #the no. of rows in the image are less than 30
        fd=0
        for j in range(t[0]-1,-1,-1):
            for i in range(0,t[1]):
                if skel[j][i]==255:
                    ab3=i
                    ab2=j
                    fd=1
                    break
            if fd==1:
                break 
        base=t[0]
        pi=22/7
        max1=-9999
        if t[1]>ab3+45:
            c=45
        else:
            c=abs(t[1]-ab3)
        for i in range(0,c):
             m=math.atan2(i,base)
             degree = m*(180/pi)
             slope=i/base
             y1=ab3+i
             count=0
             for x_final in range(ab1,ab2+1):
                 y_final=int(slope*(ab1-x_final)+y1)
                 if skel[x_final][y_final]==255:
                     count+=1
             if max1<count:
                 max1=count
                 deg_final=degree
                 yy=y1
                 slope1=slope
        skel1=slant(skel,deg_final,ab2)    
    return skel1

#FOR REMOVING THE EXTRA SPACE AROUND THE ALPHABETS JUST BEFORE PASSING IT TO THE NEURAL NETWORK IN ORDER TO CLASSIFY THEM CORRECTLY
def crop(a):
    t=a.shape
    count=0
    f=0
    for i in range(0,t[0]):
        for j in range(0,t[1]):
            if a[i][j]!=0:
                count+=1
                t1=i
                f=1
                break
        if f==1:
            break
        
    if count==0:
        return 0
    f=0
    for i in range(0,t[1]):
        for j in range(0,t[0]):
            if a[j][i]!=0:
                t2=i
                f=1
                break
        if f==1:
            break
    f=0
    for i in range(t[0]-1,-1,-1):
        for j in range(0,t[1]):
            if a[i][j]!=0:
                t3=i
                f=1
                break
        if f==1:
            break
    f=0
    for i in range(t[1]-1,-1,-1):
        for j in range(0,t[0]):
            if a[j][i]!=0:
                t4=i
                f=1
                break
        if f==1:
            break
    if t1>0:
        t1-=1
    if t2>0:
        t2-=1
    if t3<t[0]-1:
        t3+=1
    if t4<t[1]-1:
        t4+=1

    rows=t3-t1+1
    colm=t4-t2+1
    a1=np.zeros((rows,colm), dtype=np.uint8)
    r=0
    c=0
    for i in range(t1,t3+1):
        c=0
        for j in range(t2,t4+1):
            a1[r][c]=a[i][j]
            c+=1
        r+=1
    return a1

#THE FUNCTION CLASSIFIES THE INPUT ALPHABET INTO ITS PROPER CATEGORY 
def cnn(s1,kkt):
    #classification layer
    ret,b = cv2.threshold(s1,127,255,cv2.THRESH_BINARY)
    b=crop(b)
    kernel = np.ones((5,5), np.uint8)
    kernel1 = np.ones((3,3), np.uint8)
    dilation = cv2.dilate(b,kernel,iterations = 1)
    erosion = cv2.erode(dilation,kernel1,iterations = 1)
    newImage = cv2.resize(erosion, (28, 28))
    newImage = np.array(newImage)
    newImage = newImage.astype('float32')/255
    prediction2 = cnn_model.predict(newImage.reshape(1,28,28,1))[0]
    prediction2 = np.argmax(prediction2)
    fig.add_subplot(2,10,kkt)
    plt.title(str(letters[int(prediction2)+1]))
    plt.imshow(newImage,cmap='gray')
    return str(letters[int(prediction2)+1])

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import load_model
from collections import deque
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the model built in the previous steps
cnn_model = load_model('emnist_cnn_model.h5')

# Letters lookup
letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}

fig=plt.figure()
fig.tight_layout()
kt=1
for i1 in range(2,3):
    st=""
    #input any word image to be classified with font color of word to be darker than that of the background
    s="a03-006-00-0"+str(i1)+".png"
    print(s)
    
    # Read image
    img = cv2.imread(s, cv2.IMREAD_GRAYSCALE);
    ret2,thresh1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    fig.add_subplot(2,10,kt)
    plt.title(st)
    plt.imshow(img,cmap='gray')
    kt+=1

    #calculate angle and remove slant
    skel=angle(thresh1)
    
    #skel=thresh1
    t=skel.shape
    for i in range(0,t[0]):
        for j in range(0,t[1]):
            if skel[i][j]==150 or skel[i][j]==255:
                skel[i][j]=255
    skel1=skeletonize(skel) 

    #find the potential segmentation points
    t=skel1.shape
    for i in range(0,t[0]):
        for j in range(0,t[1]):
            if skel1[i][j]==255 and i>0 and j>0 and i<t[0]-1 and j<t[1]-1:
                if skel1[i-1][j]==0 and skel1[i+1][j]==0 and skel1[i][j+1]==0 and skel1[i][j-1]==0 and skel1[i-1][j-1]==0 and skel1[i+1][j+1]==0 and skel1[i-1][j+1]==0 and skel1[i+1][j-1]==0:
                    skel1[i][j]=0
    
    k=[]
    for i in range(0,int(t[1])):
        sum1=0
        for j in range(0,int(t[0])):
            if skel1[j][i]==255:
                sum1+=1
        if sum1<=1:
            k.append(i)       
    sum1=0
    count=0
    k1=[]
    
    #draw segmentation lines
    for i in range(0,len(k)-1):
        if sum1==0:
            sum1+=k[i]
            count+=1
        #take average of lines at a distance of 7 or less    
        if k[i+1]<=k[i]+7:
            sum1+=k[i+1]
            count+=1
        else:
            if count>=3:
                th=int(sum1/count)
                k1.append(int(th))
                sum1=0
                count=0
                

    #actual segmentation of words at precalculated segmentation points and classification of each alphabet
    k2=0
    for i in k1:
        s1 = skel1[:, k2:i]
        if s1.shape[0]==0 or s1.shape[1]==0:
            continue
        st+=str(cnn(s1,kt))
        kt+=1
        k2=i
    s1 = skel1[:, k2:]
    i+=1
    st+=str(cnn(s1,kt))
    print()
    

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
