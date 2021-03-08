
# base
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float, exposure
from skimage import io, exposure, measure                 # import images, extract measurements from an image
from skimage.color import rgb2grey                      # Tools to convert images to grayscale
from skimage.filters import threshold_otsu, threshold_yen, try_all_threshold  # To detect a threshold value for binarizing

# https://drive.google.com/drive/u/1/folders/1iIznoItdc160ge6uUFaqv9rb9Lw2SqN_
# change folder location
folder="F:\sps python\cell_img\\"


# python "F:\sps python\plt_img3.py"
def get_points(file,num,fig, ax):
    image = plt.imread(folder+file)
    img = img_as_float(image)
    img_grey = rgb2grey(img)
    # t = threshold_otsu(img_grey)
    t = threshold_yen(img_grey)
    # print("t:",t)   #0.005
    img_binarised = img_grey > t  #<
    img_labelled = measure.label(img_binarised.astype('uint8'))
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    ax[num][0].imshow(img)
    # ax[num][0].set_title('original', fontsize=12)
    ax[num][0].set_title(file, fontsize=12)
    ax[num][1].imshow(img_adapteq)
    ax[num][1].set_title('brighten', fontsize=12)
    ax[num][2].imshow(img_labelled, cmap='gray')
    ax[num][2].set_title('locate cells', fontsize=12)

    info = measure.regionprops(img_labelled)
    no_of_regions = len(info)
    # print("no_of_regions",no_of_regions)

    pts=np.array([[0,0]])
    bright=np.array([])
    for i in range(no_of_regions):
        x, y = info[i].centroid
        x,y=int(x),int(y)
        a=info[i].area
        # print(f'c{info[i].centroid} a{info[i].area}')
        if a>5:
        # if a>5 and img_grey[x,y]>0.05:
            # ax[num][2].text(y, x, info[i].area, ha='center',color='white',fontsize=8)
            ax[num][2].text(y, x, '.', ha='center',color='red',fontsize=12)
            pts=np.append(pts,np.array([[x,y]]),0)
            bright=np.append(bright,np.array(img_grey[x,y]))
            pass

    print(len(pts),pts.shape)
    # return pts[1:],bright[1:]
    return pts[1:],bright

def twocol(file,file2,cpts,cbright):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 4))
    pts1,bright1=get_points(file,0,fig, ax)
    pts2,bright2=get_points(file2,1,fig, ax)
    pts1r=np.around(pts1)
    pts2r=np.around(pts2)
    for i,[x,y] in enumerate(pts1r):
        if [x,y] in pts2r:
            # cbright=np.append(cbright,[bright1[i]],0)
            cbright=np.append(cbright,bright1[i])
            cpts=np.append(cpts,np.array([[x,y]]),0)
    return cpts,cbright

def threeimg(i,j):
    cpts=np.array([[0,0]])
    cbright=np.array([])
    for x in range(3):
        file1=a[i]+b[j]+c[0]+d[2*i+j][x]+".jpg"
        file2=a[i]+b[j]+c[1]+d[2*i+j][x]+".jpg"
        print(file1)
        cpts,cbright=twocol(file1,file2,cpts,cbright)
        # normalize
        # mean=np.mean(cbright)
        print('found',len(cbright),'cells')
        fig=plt.figure(figsize=(5, 3))
        plt.scatter(cbright, np.array([0]*len(cbright)))
        # fig.set_xlabel('cell brightness')
        plt.xlabel('cell brightness')
        plt.show()
    return cpts[1:],cbright


a=['M22','RP22']
b=['+IPTG ','-IPTG ']
c=['CFP ','YFP ']
d=['2 400ms','3 400ms','4 400ms'],['1','2 400ms','3 400ms'],['1','2','3'],['3','4','5']
def run():

    # go through M22+IPTG CFP/YFP 2/3/4 400ms.jpg
    # i=0
    # j=0
    # cpts,cbright=threeimg(i,j)

    # go through all images
    for i in [0,1]:
        for j in [0,1]:
            cpts,cbright=threeimg(i,j)

    return

run()
