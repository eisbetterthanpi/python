
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float, exposure
from skimage import io, exposure, measure
from skimage.color import rgb2grey
from skimage.filters import threshold_otsu, threshold_yen, try_all_threshold

# https://drive.google.com/drive/u/1/folders/1iIznoItdc160ge6uUFaqv9rb9Lw2SqN_
# change file location
file="F:\sps python\cell_img\M22+IPTG YFP 4 400ms.jpg"

# python "F:\sps python\plt_img2.py"
image = plt.imread(file)
img = img_as_float(image)
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

img_grey = rgb2grey(img)
# t = threshold_otsu(img_grey)
t = threshold_yen(img_grey)
# print("threshold:",t)   #0.005
img_binarised = img_grey > t
img_labelled = measure.label(img_binarised.astype('uint8'))

info = measure.regionprops(img_labelled)
no_of_regions = len(info)

# for prop in info[0]:
#      print('-'*20, prop, '-'*20)
#      print(info[0][prop],'\n')

bright=np.array([])
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 4))
for i in range(no_of_regions):
    x, y = info[i].centroid
    x,y=int(x),int(y)
    a=info[i].area
    if a>5: #take circles with area > 5
    # if a>5 and img_grey[x,y]>0.05:
        # ax[2].text(y, x, round(img_grey[x,y],2), ha='center',color='white',fontsize=8)
        ax[2].text(y, x, '.', ha='center',color='red',fontsize=12)
        bright=np.append(bright,np.array(img_grey[x,y]))

ax[0].imshow(img)
ax[0].set_title('original')
ax[1].imshow(img_labelled, cmap='gray')
ax[1].set_title(f'Binarise')
ax[2].imshow(img_labelled, cmap='gray')
ax[2].set_title('locate cells')
ax[3].imshow(img_adapteq)
ax[3].set_title('brighten')

plt.show()

print('found',len(bright),'cells')
fig=plt.figure(figsize=(5, 3))
plt.scatter(bright, np.array([0]*len(bright)))
# fig.set_xlabel('cell brightness')
plt.xlabel('cell brightness')
plt.show()
