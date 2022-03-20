import cv2
import numpy as np

FILE_NAME = 'F:\\projection\\album background\\young and free 3 order.jpg'
#FILE_NAME = 'F:\\media\\fopx logo.jpg'
img = cv2.imread(FILE_NAME,0)
a=[]
cv2.imshow('image',img)

#print(img.shape)
height, width = img.shape[:2]
h,w=18,25
#res = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation = cv2.INTER_CUBIC)
img = cv2.resize(img, (int(w), int(h)), interpolation = cv2.INTER_CUBIC)
#cv2.INTER_CUBIC cv2.INTER_AREA cv2.INTER_LINEAR
##res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
##res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

#cv2.imshow('resized',img)
#cv2.imwrite('F:\\projection\\album background\\young and free 3 res.jpg', img)

#print(img[18,24])
#print(res.shape)

#blue = img[100,100,0]
#img.item(10,10,2)
#img.itemset((10,10,2),100)

height, width = img.shape[:2]
for y in range(height):
    for x in range(width):
        #print(y,x,img[y,x])
        if img[y,x] < 25:
            a.append(chr(9608))
            #print("1")
        elif img[y,x] < 70:
            a.append(chr(9619))
            #print("2")
        elif img[y,x] < 130:
            a.append(chr(9618))
            #print("3")
        elif img[y,x] < 200:
            a.append(chr(9617))
            #print("4")
        else:
            a.append(chr(10240))
            a.append(' ')
            #print("5")
        #[10240,9617,9618,9619,9608]
        #a.append(chr(n))
    ans=''.join(j for j in a)
    print(ans)
    a,ans=[],[]


##ans=''.join(j for j in a)
##print(ans)

##for x in [10240,9617,9618,9619,9608]:
##    print(chr(x),end='')




