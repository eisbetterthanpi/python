
# base
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float, exposure
from skimage import io, exposure, measure                 # import images, extract measurements from an image
from skimage.color import rgb2grey                      # Tools to convert images to grayscale
from skimage.filters import threshold_otsu, threshold_yen, try_all_threshold  # To detect a threshold value for binarizing

# https://drive.google.com/drive/u/1/folders/1iIznoItdc160ge6uUFaqv9rb9Lw2SqN_
# change folder location
# folder="F:\sps python\cell_img\\"
folder="F:\sps python\jpged\\"

# python "F:\sps python\plt_all.py"
def get_points(file,num,fig, ax):
    image = plt.imread(folder+file)
    img = img_as_float(image)
    img_grey = rgb2grey(img)
    # t = threshold_otsu(img_grey)
    t = threshold_yen(img_grey)
    # print("t:",t)   #0.005
    # img_binarised = img_grey > t  #<
    img_binarised = img_grey < t
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
            # pts=np.append(pts,np.array([[y,x]]),0)
            bright=np.append(bright,np.array(img_grey[x,y]))
            pass

    # print(len(pts),pts.shape)
    return pts[1:],img_grey

def draw(file,num,fig, ax):
    image = plt.imread(folder+file)
    img = img_as_float(image)
    img_grey = rgb2grey(img)
    # t = threshold_otsu(img_grey)
    t = threshold_yen(img_grey)
    # print("t:",t)   #0.005
    # img_binarised = img_grey > t  #<
    # img_binarised = img_grey < t
    # img_labelled = measure.label(img_binarised.astype('uint8'))
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    ax[num][0].imshow(img)
    # ax[num][0].set_title('original', fontsize=12)
    ax[num][0].set_title(file, fontsize=12)
    ax[num][1].imshow(img_adapteq)
    ax[num][1].set_title('brighten', fontsize=12)
    # ax[num][2].imshow(img_labelled, cmap='gray')
    # ax[num][2].set_title('locate cells', fontsize=12)

    # info = measure.regionprops(img_labelled)
    # no_of_regions = len(info)
    # print("no_of_regions",no_of_regions)

    # pts=np.array([[0,0]])
    # bright=np.array([])
    # for i in range(no_of_regions):
    #     x, y = info[i].centroid
    #     x,y=int(x),int(y)
    #     a=info[i].area
    #     # print(f'c{info[i].centroid} a{info[i].area}')
    #     if a>5:
    #     # if a>5 and img_grey[x,y]>0.05:
    #         # ax[num][2].text(y, x, info[i].area, ha='center',color='white',fontsize=8)
    #         ax[num][2].text(y, x, '.', ha='center',color='red',fontsize=12)
    #         pts=np.append(pts,np.array([[x,y]]),0)
    #         # pts=np.append(pts,np.array([[y,x]]),0)
    #         bright=np.append(bright,np.array(img_grey[x,y]))
    #         pass
    return img_grey


# def twocol(bf,cpts):
def twocol(bf,file1,file2,cpts):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 4))
    pts,img=get_points(bf,0,fig, ax)
    ptsr=np.around(pts)
    # union of points
    # cpts=np.unique(np.append(pts1r,pts2r,0),axis=0)
    # print("test here",cpts)
    # print(len(cpts),cpts.shape)
    # print(img1.shape)

    img1=draw(file1,0,fig, ax) # gray
    img2=draw(file2,1,fig, ax)

    nb1=np.array([])
    nb2=np.array([])
    for x,y in ptsr:
        # print(x,y,img1[x,y])
        nb1=np.append(nb1,img1[x,y])
        nb2=np.append(nb2,img2[x,y])
    # # nb1=np.std(img1[cpts])
    #
    nb1=(nb1-np.mean(nb1))/np.std(nb1) # ok, around 0, got stray at high
    nb2=(nb2-np.mean(nb2))/np.std(nb2)
    # nb2=(nb2)/np.std(nb2)
    # nb2=(nb2)/np.std(nb2)

    # fig=plt.figure(figsize=(5, 3))
    # plt.scatter(nb1, np.array([0]*len(nb1)))
    # plt.xlabel('cell brightness')
    # plt.show()
    # plt.scatter(cb1, cb2)
    ax[1][2].scatter(nb1, nb2)
    ax[1][2].set_title('normalised')
    plt.close()
    # return cpts,nb1,nb2
    return ptsr,nb1,nb2


def threeimg(i,j):
    cpts=np.array([[0,0]])
    cb1=np.array([])
    cb2=np.array([])
    # for x in range(3):
    for x in range(len(d[2*i+j])):
        bf=a[i]+b[j]+c[0]+d[2*i+j][x]+".jpg"
        file1=a[i]+b[j]+c[1]+d[2*i+j][x]+".jpg"
        file2=a[i]+b[j]+c[2]+d[2*i+j][x]+".jpg"
        # print(file1)
        cpts,nb1,nb2=twocol(bf,file1,file2,cpts)
        # cpts,nb1,nb2=twocol(file1,file2,cpts)
        # normalize
        cb1=np.append(nb1,cb1)
        cb2=np.append(nb2,cb2)
        print(a[i]+b[j]+d[2*i+j][x],'found',len(cpts),'cells')
        # fig=plt.figure(figsize=(5, 3))
        # plt.scatter(cbright, np.array([0]*len(cbright)))
        # # fig.set_xlabel('cell brightness')
        # plt.xlabel('cell brightness')
        # plt.show()
    # return cpts[1:],cbright
    return cb1,cb2

# n2int=
a=['M22 ','RP22 ']
b=['+IPTG ','-IPTG ']
c=['BF ','CFP ','YFP ']
d=[['1','4','5','6'],['1','2','3','4'],['1','2','3','4'],['1','2','3','4','5','6']]
# python "F:\sps python\pltfin.py"
def run():

    # go through M22+IPTG CFP/YFP 2/3/4 400ms.jpg
    i=0
    j=0
    # cb1,cb2=threeimg(i,j)
    # fig=plt.figure(figsize=(5, 3))
    # plt.scatter(cb1, cb2)
    # # fig.set_xlabel('cell brightness')
    # plt.xlabel('chaaess')
    # plt.show()

    # go through all images, plot graph
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 4))
    for i in [0,1]:
        for j in [0,1]:
            cb1,cb2=threeimg(i,j)
            # fig=plt.figure(figsize=(5, 3))
            ax[i,j].scatter(cb1, cb2,1)
            ax[i,j].set_title(a[i]+b[j])
            plt.xlabel(c[1])
            plt.ylabel(c[2])
    plt.show()
    return

run()
