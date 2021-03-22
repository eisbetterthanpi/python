
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float, exposure
from skimage import io, exposure, measure                 # import images, extract measurements from an image
from skimage.color import rgb2grey                      # Tools to convert images to grayscale
from skimage.filters import threshold_otsu, threshold_yen, try_all_threshold  # To detect a threshold value for binarizing
from skimage.filters import gaussian

# https://drive.google.com/drive/u/1/folders/1iIznoItdc160ge6uUFaqv9rb9Lw2SqN_
# change folder location
folder="F:\sps python\jpged\\" # only jpeg images

# python "F:\sps python\pltfin.py"
def get_points(file,num,fig, ax):
    image = plt.imread(folder+file)
    img = img_as_float(image)
    img_grey = rgb2grey(img)
    # t = threshold_otsu(img_grey)
    t = threshold_yen(img_grey)
    # print("t:",t)   #0.005
    img_binarised = img_grey < t
    img_labelled = measure.label(img_binarised.astype('uint8'))
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    ax[num][0].imshow(img)
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
            # ax[num][2].text(y, x, info[i].area, ha='center',color='white',fontsize=8)
            ax[num][2].text(y, x, '.', ha='center',color='red',fontsize=12)
            pts=np.append(pts,np.array([[x,y]]),0)
            bright=np.append(bright,np.array(img_grey[x,y]))
            pass
    return pts[1:],img_grey

def draw(file,num,fig, ax):
    image = plt.imread(folder+file)
    img = img_as_float(image)
    img_grey = rgb2grey(img)
    # t = threshold_otsu(img_grey)
    t = threshold_yen(img_grey)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    img_blur=gaussian(img, sigma=3) #blur for plot only, not for analysis

    ax[num][0].imshow(img)
    ax[num][0].set_title(file, fontsize=12)
    # ax[num][1].imshow(img_adapteq) #choose image to show in top mid
    ax[num][1].imshow(img_blur)
    ax[num][1].set_title('blur', fontsize=12)
    return img_grey

def twocol(bf,file1,file2,cpts):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 4))
    pts,img=get_points(bf,0,fig, ax)
    ptsr=np.around(pts)

    img1=draw(file1,0,fig, ax) # gray
    img2=draw(file2,1,fig, ax)
    # https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian
    img1=gaussian(img1, sigma=3) #change sigma for actual analysis
    img2=gaussian(img2, sigma=3)

    nb1=np.array([])
    nb2=np.array([])
    for x,y in ptsr:
        # print(x,y,img1[x,y])
        nb1=np.append(nb1,img1[x,y])
        nb2=np.append(nb2,img2[x,y])
    # # nb1=np.std(img1[cpts])

    # normalize
    # nb1=(nb1-np.mean(nb1))/np.std(nb1) # ok, around 0, got stray at high
    # nb2=(nb2-np.mean(nb2))/np.std(nb2)
    # nb1=(nb1)/np.std(nb1)/np.mean(nb1)
    # nb2=(nb2)/np.std(nb2)/np.mean(nb2)
    nb1=(nb1)/np.mean(nb1)
    nb2=(nb2)/np.mean(nb2)

    ax[1][2].scatter(nb1, nb2,1)
    ax[1][2].set_title('normalised')
    plt.close() #comment out to show analysis of each cfp gfp img pair
    return ptsr,nb1,nb2

def threeimg(i,j):
    cpts=np.array([[0,0]])
    cb1=np.array([])
    cb2=np.array([])
    for x in range(len(d[2*i+j])):
        bf=a[i]+b[j]+c[0]+d[2*i+j][x]+".jpg"
        file1=a[i]+b[j]+c[1]+d[2*i+j][x]+".jpg"
        file2=a[i]+b[j]+c[2]+d[2*i+j][x]+".jpg"
        # print(file1)
        cpts,nb1,nb2=twocol(bf,file1,file2,cpts)

        cb1=np.append(nb1,cb1)
        cb2=np.append(nb2,cb2)
        print(a[i]+b[j]+d[2*i+j][x],'found',len(cpts),'cells')
        # plt.show() #uncomment to show each cfp gfp img pairs
    return cb1,cb2


a=['M22','RP22']
# a=['M22 ','RP22 ']
b=['+IPTG ','-IPTG ']
c=['BF ','CFP ','YFP ']
d=['2 400ms','3 400ms','4 400ms'],['1','2 400ms','3 400ms'],['1','2','3'],['3','4','5']
# d=[['1','4','5','6'],['1','2','3','4'],['1','2','3','4'],['1','2','3','4','5','6']]
# python "F:\sps python\pltfinb.py"
def run():

    # go through M22+IPTG CFP/YFP 2/3/4 400ms.jpg
    # i=0 #0:M22 1:RP22
    # j=0 #0:+IPTG 1:-IPTG
    # cb1,cb2=threeimg(i,j)
    # fig=plt.figure(figsize=(5, 3))
    # plt.scatter(cb1, cb2,1)
    # plt.xlabel('chaaess')
    # plt.show()

    # go through all images, plot graph
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 4))
    for i in [0,1]:
    # for i in [0]:
        for j in [0,1]:
        # for j in [0]:
            cfp,yfp=threeimg(i,j)
            # fig=plt.figure(figsize=(5, 3))
            ax[i,j].scatter(cfp, yfp,1)
            ax[i,j].set_title(a[i]+b[j])
            # plt.xlabel(c[1])
            # plt.ylabel(c[2])
            nint=(np.mean((cfp-yfp)**2)/(2*np.mean(cfp)*np.mean(yfp)))**(1/2)
            next=((np.mean(cfp*yfp)-np.mean(cfp)*np.mean(yfp))/(np.mean(cfp)*np.mean(yfp)))**(1/2)
            ntot=((np.mean((cfp**2)+(yfp**2))-2*np.mean(cfp)*np.mean(yfp))/(2*np.mean(cfp)*np.mean(yfp)))**(1/2)
            dp=3
            nint,next,ntot=np.round(nint,dp),np.round(next,dp),np.round(ntot,dp)
            print(nint,next,ntot)
    plt.show()

    return

run()
