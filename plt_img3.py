
# base
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float, exposure
from skimage import io, exposure, measure                 # import images, extract measurements from an image
from skimage.color import rgb2grey                      # Tools to convert images to grayscale
from skimage.filters import threshold_otsu, threshold_yen, try_all_threshold  # To detect a threshold value for binarizing

# https://drive.google.com/drive/u/1/folders/1iIznoItdc160ge6uUFaqv9rb9Lw2SqN_
folder="F:\sps python\cell_img\\" #F:\sps python\cell_img
file="F:\sps python\cell_img\M22+IPTG CFP 3 400ms.jpg"
file2="F:\sps python\cell_img\M22+IPTG YFP 3 400ms.jpg"



# python "F:\sps python\plt_img3.py"

def get_points(file,num,fig, ax):

    pts=np.array([[0,0]])
    # bright=np.array([[0,0]])
    bright=np.array([])
    # print("get pt",lp)
    print(pts)

    image = plt.imread(file)
    img = img_as_float(image)
    img_grey = rgb2grey(img)
    # print("col",img_grey.shape)
    # t = threshold_otsu(img_grey)
    t = threshold_yen(img_grey)
    print("t:",t)   #0.005
    img_binarised = img_grey > t  #<
    img_labelled = measure.label(img_binarised.astype('uint8'))
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    ax[num][0].imshow(img)
    ax[num][0].set_title('original', fontsize=12)
    ax[num][1].imshow(img_adapteq)
    ax[num][1].set_title('img_adapteq', fontsize=12)
    ax[num][2].imshow(img_labelled, cmap='gray')
    ax[num][2].set_title('img_labelled', fontsize=12)

    info = measure.regionprops(img_labelled)
    no_of_regions = len(info)
    print("no_of_regions",no_of_regions)

    for i in range(no_of_regions):
        x, y = info[i].centroid
        x,y=int(x),int(y)
        # print(f'({x},{y})')
        a=info[i].area
        # print(f'c{info[i].centroid} a{info[i].area}')
        if a>5:
            # ax[0][1].text(y, x, f'{x,y}', ha='center',color='white')
            ax[num][2].text(y, x, info[i].area, ha='center',color='white',fontsize=8)
            # print([x,y])
            pts=np.append(pts,np.array([[x,y]]),0)
            # print('img_grey[x,y]',img_grey[x,y])
            # bright=np.append(bright,np.array([img_grey[x,y]]),0)
            bright=np.append(bright,np.array(image[x,y]),0)
            # pts=np.append(pts,[[x,y]],0)
            pass

    print(len(pts),pts.shape)
    return pts[1:],bright[1:]

def twocol(file,file2,cpts,cbright):
    lp=9
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 4))
    print("2col")
    pts1,bright1=get_points(file,0,fig, ax)
    pts2,bright2=get_points(file2,1,fig, ax)
    # np.around(pst2, decimals=0, out=None)
    pts1r=np.around(pts1)
    pts2r=np.around(pts2)
    # cpts=np.array([[0,0]])
    # cbright=np.array([])
    for i,[x,y] in enumerate(pts1r):
        if [x,y] in pts2r:
            print(i)
            cbright=np.append(cbright,[bright1[i]],0)
            # ax[1][2].text(y, x, 0, ha='center',color='white',fontsize=8)
            cpts=np.append(cpts,np.array([[x,y]]),0)

    # for x in range(len(pts1r)):
    #     print(pts1r[x],' ',pts2r[x])
    # np.setdiff1d(

    # print(len(pts1r),len(pts2r),len(cpts))
    # print(cpts,len(cpts))
    # return cpts,cbright[1:]
    return cpts,cbright

def threeimg(i,j):
    cpts=np.array([[0,0]])
    cbright=np.array([])
    print("3img")
    for x in range(3):
    # for x in range(1,2):
        file1=folder+a[i]+b[j]+c[0]+d[2*i+j][x]+".jpg"
        file2=folder+a[i]+b[j]+c[1]+d[2*i+j][x]+".jpg"
        print(file1)
        cpts,cbright=twocol(file1,file2,cpts,cbright)
        # normalize
        # mean=np.mean(cbright)
        plt.figure(figsize=(10, 5))
        plt.scatter(cbright, np.array([0]*len(cbright)))
        # np.linalg.norm()
        plt.show()

    return cpts,cbright




a=['M22','RP22']
b=['+IPTG ','-IPTG ']
c=['CFP ','YFP ']
d=['2 400ms','3 400ms','4 400ms'],['1','2 400ms','3 400ms'],['1','2','3'],['3','4','5']
def run():
    # '400ms'
    print("run")
    # ['2 400ms','3 400ms','4 400ms']
    # ['1','2 400ms','3 400ms']
    # ['1','2','3']
    # ['3','4','5']
    i=0
    j=0
    cpts,cbright=threeimg(i,j)

    # file=folder+a[]+b[]+c[]+d[][]
    return

run()



# image = plt.imread(file)
# img = img_as_float(image)
# img_grey = rgb2grey(img)
# print('greyshape',img_grey.shape)
# print(img_grey[100,48])
# for x,y in cpts:
#     x=int(x)
#     y=int(y)
#     # print(x,' ',y)
#     b=img_grey[x,y]
#     if b>0:
#         # ax[0][1].text(y, x, f'{x,y}', ha='center',color='white')
#         ax[1][1].text(y, x, f'x,y', ha='center',color='black',fontsize=8)
#         # print([x,y])
#         # pts=np.append(pts,[[x,y]],0)
#         pass



# ax[0][0].imshow(img)
# ax[0][0].set_title('jh', fontsize=12)
# ax[0][1].imshow(img_labelled, cmap='gray')
# ax[0][1].set_title('img_labelled', fontsize=12)
#
# ax[0][2].imshow(img_labelled, cmap='gray')


# plt.tight_layout()
# # plt.savefig('09_image-processing-1_four-circles-layers.png', dpi=150)
# plt.show()



# normalize
# from sklearn.preprocessing import normalize
# x = np.random.rand(1000)*10
# norm1 = x / np.linalg.norm(x)
# norm2 = normalize(x[:,np.newaxis], axis=0).ravel()
# print np.all(norm1 == norm2)
