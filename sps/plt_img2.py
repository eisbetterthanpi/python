
# base
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float, exposure
from skimage import io, exposure, measure                 # import images, extract measurements from an image
from skimage.color import rgb2grey                      # Tools to convert images to grayscale
from skimage.filters import threshold_otsu, threshold_yen, try_all_threshold  # To detect a threshold value for binarizing

# https://drive.google.com/drive/u/1/folders/1iIznoItdc160ge6uUFaqv9rb9Lw2SqN_
# file="F:\sps python\cell_img\M22+IPTG CFP 4 400ms.jpg"
file="F:\sps python\cell_img\M22+IPTG YFP 2 400ms.jpg"

# python "F:\sps python\plt_img2.py"

pts=np.array([])
# def get_points(file):
image = plt.imread(file)

img = img_as_float(image)
p2, p98 = np.percentile(img, (0.2, 100))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
# img=img_rescale
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
# img=img_adapteq

# img=image
img_grey = rgb2grey(img)
# t = threshold_otsu(img_grey)
t = threshold_yen(img_grey)
print("t:",t)   #0.005
img_binarised = img_grey > t  #<
img_labelled = measure.label(img_binarised.astype('uint8'))
# img_labelled=img


info = measure.regionprops(img_labelled)         # Extract all the properties of the labelled regions
no_of_regions = len(info)
print("no_of_regions",no_of_regions)
# for prop in info[0]:
#      print('-'*20, prop, '-'*20)
#      print(info[0][prop],'\n')

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 4))
for i in range(no_of_regions):
    x, y = info[i].centroid
    print(f'({x},{y})')
    a=info[i].area
    # print(f'c{info[i].centroid} a{info[i].area}')
    if a>5:
        # ax[1].text(y, x, f'{i}({info[i].area})', ha='center',color='white')
        ax[0][2].text(y, x, info[i].area, ha='center',color='white',fontsize=8)
        # pts+=(x,y)
        pass
# p
# return pts

# pts=get_points(file)


# ax[0].imshow(img)
# ax[0].set_title('jh', fontsize=12)
# ax[1].imshow(img_labelled, cmap='gray')
# ax[1].set_title(f'Binarised (Using Yen)', fontsize=12)

ax[0][0].imshow(img)
ax[0][0].set_title('jh', fontsize=12)
ax[0][1].imshow(img_labelled, cmap='gray')
ax[0][1].set_title('img_labelled', fontsize=12)
ax[0][2].imshow(img_labelled, cmap='gray')

ax[0][0].imshow(img)
ax[1][0].imshow(img_rescale, cmap='gray')
ax[1][1].imshow(img_adapteq, cmap='gray')


# ax[2].imshow(img, cmap='jet')
# ax[2].set_title(f'Labelled Objects', fontsize=12)
# for a in ax.flat:
#     a.axis('off')

bins=256
# ax_img, ax_hist = axes[:, 0]
ax_hist=ax[1][2]
ax_cdf = ax_hist.twinx() #overlay graphs
# Display histogram
ax_hist.hist(img_adapteq.ravel(), bins=bins, histtype='step', color='black')
ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax_hist.set_xlabel('Pixel intensity')
ax_hist.set_xlim(0, 1)
ax_hist.set_yticks([])
# Display cumulative distribution
img_cdf, bins = exposure.cumulative_distribution(image, bins)
ax_cdf.plot(bins, img_cdf, 'r')
ax_cdf.set_yticks([])
y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))
ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 5))


plt.tight_layout()
# plt.savefig('09_image-processing-1_four-circles-layers.png', dpi=150)
plt.show()
