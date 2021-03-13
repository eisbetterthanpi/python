
# base
import matplotlib.pyplot as plt
from skimage import io, exposure, measure                 # import images, extract measurements from an image
from skimage.color import rgb2grey                      # Tools to convert images to grayscale
from skimage.filters import threshold_otsu, threshold_yen, try_all_threshold  # To detect a threshold value for binarizing

# # img = plt.imread('./four-circles.jpg')    # Load the image
# file="C:\\Users\\matth\\Pictures\\Screenshots\\Screenshot (16).png" #both can
# file="C:/Users/matth/Pictures/Screenshots/Screenshot (16).png"
# file="F:\insta\\boring girlfriend that hurt.jpg"

# https://drive.google.com/drive/u/1/folders/1iIznoItdc160ge6uUFaqv9rb9Lw2SqN_
file="F:\sps python\M22+IPTG CFP 3 400ms.jpg"
# file="F:\sps python\M22+IPTG YFP 3 400ms.jpg"

# python "F:\sps python\plt_img.py"
image = plt.imread(file)
# plt.axis('off')
# plt.imshow(img)   # plt.show()
# print(img.shape)

# # plt.style.use('ggplot')
# r_data = img[:, :, 0].flatten() # Convert the 2D array to a long 1D list using 'flatten'
# g_data = img[:, :, 1].flatten()        # The flattend GREEN channel
# b_data = img[:, :, 2].flatten()        # The flattend BLUE channel
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
# ax[0].hist(r_data, color='red', bins=range(0, 64), density=True)
# ax[1].hist(g_data, color='green', bins=range(0, 64), density=True)
# ax[2].hist(b_data, color='blue', bins=range(0, 64), density=True)

# ax[1].imshow(img[:, :, 0], cmap='gray')
# ax[1].imshow(img[:, :, 0], cmap='Reds')
# ax[1].set_title('Red Channel(in Grayscale)')
# ax[2].imshow(img[:, :, 1], cmap='Greens')
# ax[2].set_title('Green Channel(in Grayscale)')
# ax[3].imshow(img[:, :, 2], cmap='Blues')
# ax[3].set_title('Blue Channel(in Grayscale)')

# ax[0].set_title('Red values')
# ax[1].set_title('Green values')
# ax[2].set_title('Blue values')
# plt.tight_layout()
# plt.show()



import numpy as np
from skimage import data, img_as_float, exposure


# image=img_labelled
img = img_as_float(image)
p2, p98 = np.percentile(img, (0.0, 20))
# p2, p98 = np.percentile(img, (0.2, 100))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98)) # Contrast stretching
# img_eq = exposure.equalize_hist(img) #linear, ups noise
# img=img_rescale
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
# img=img_adapteq
img_rescale = exposure.rescale_intensity(img_adapteq, in_range=(p2, p98)) # Contrast stretching


# img=image
img_grey = rgb2grey(img) #rgb2grey(img) # img_grey = rgb2gray(rgba2rgb(img))
# img = img_as_float(img)
# t = threshold_otsu(img_grey)
t = threshold_yen(img_grey)
print("t:",t)   #0.005
img_binarised = img_grey > t  #<
# img_binarised = img_grey > 0.05 #0.05
img_labelled = measure.label(img_binarised.astype('uint8'))
# img_labelled=img

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 4))

# ax[0].imshow(img)
# ax[0].set_title('jh', fontsize=12)
# ax[1].imshow(img_labelled, cmap='gray')
# ax[1].set_title(f'Binarised (Using Yen)', fontsize=12)

ax[0][0].imshow(img)
ax[0][0].set_title('jh', fontsize=12)
ax[0][1].imshow(img_labelled, cmap='gray')
ax[0][1].set_title('img_labelled', fontsize=12)

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

info = measure.regionprops(img_labelled)         # Extract all the properties of the labelled regions
no_of_regions = len(info)
print("no_of_regions",no_of_regions)
# for i in range(no_of_regions):
#     #print('-'*10, f'Region {i}', '-'*10)
#     #print(info[i].centroid,info[i].area,'\n')
#     x, y = info[i].centroid
#     # ax.text(y, x, f'Region {i}\n(Area: {info[i].area})', ha='center')
#     ax[1].text(y, x, f'Region {i}\n(Area: {info[i].area})', ha='center')
# for prop in info[0]:
#      print('-'*20, prop, '-'*20)
#      print(info[0][prop],'\n')

for i in range(no_of_regions):
    x, y = info[i].centroid
    # print(f'({x},{y})')
    a=info[i].area
    # print(f'c{info[i].centroid} a{info[i].area}')
    if a>5:
        # ax[1].text(y, x, f'{i}({info[i].area})', ha='center',color='white')
        # ax[0][1].text(y, x, info[i].area, ha='center',color='white',fontsize=8)
        pass


import numpy as np
from skimage import data, img_as_float, exposure
# matplotlib.rcParams['font.size'] = 8
img = img_as_float(image)
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98)) # Contrast stretching
# img_eq = exposure.equalize_hist(img) #linear, ups noise
img=img_rescale
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
image=img_adapteq

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
ax[0].imshow(image, cmap='gray')
# ax[0].set_title('dyjt (RGB)')
ax[0].set_title(f'Original: shape= {img.shape}', fontsize=12)

bins=256
# ax_img, ax_hist = axes[:, 0]
ax_hist=ax[1]
ax_cdf = ax_hist.twinx() #overlay graphs
# Display histogram
ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
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









# ax_img.set_axis_off()
plt.tight_layout()
# plt.savefig('09_image-processing-1_four-circles-layers.png', dpi=150)
plt.show()




# blurM = cv2.medianBlur(gray, 5)
# blurG = cv2.GaussianBlur(gray, (9, 9), 0)
# histoNorm = cv2.equalizeHist(gray)
# clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8, 8))
# claheNorm = clahe.apply(gray)
#
# def pixelVal(pix, r1, s1, r2, s2):
#     if (0 <= pix and pix <= r1):
#         return (s1 / r1) * pix
#     elif (r1 < pix and pix <= r2):
#         return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
#     else:
#         return ((255 - s2) / (255 - r2)) * (pix - r2) + s2
#
# r1 = 70
# s1 = 0
# r2 = 200
# s2 = 255
# pixelVal_vec = np.vectorize(pixelVal)
# contrast_stretched = pixelVal_vec(gray, r1, s1, r2, s2)
# contrast_stretched_blurM = pixelVal_vec(blurM, r1, s1, r2, s2)



# https://stackoverflow.com/questions/55994311/image-segmentation-to-find-cells-in-biological-images
# import cv2
# import numpy as np
# smallest_dim = min(img.shape)
# min_rad = int(img.shape[0]*0.05)
# max_rad = int(img.shape[0]*0.5) #0.5
# circles = cv2.HoughCircles((img*255).astype(np.uint8),cv2.HOUGH_GRADIENT,1,50,
#     param1=50,param2=30,minRadius=min_rad,maxRadius=max_rad)
# circles = np.uint16(np.around(circles))
# x, y, r = circles[0,:][:1][0]


# import matplotlib.pyplot as plt
# import numpy as np
# import imageio
# from skimage import data, color
# from skimage.transform import hough_circle, hough_circle_peaks
# from skimage.feature import canny
# from skimage.draw import circle_perimeter
# from skimage.util import img_as_ubyte
#
#
# gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
# gray = gray(img)
# image = np.array(gray[60:220,210:450])
# plt.imshow(image,cmap='gray')
#
# edges = canny(image, sigma=3,)
# plt.imshow(edges,cmap='gray')
#
# overlayimage = np.copy(image)
#
# # https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html
#
# hough_radii = np.arange(30, 60, 2)
# hough_res = hough_circle(edges, hough_radii)
#
# # Select the most prominent X circles
# x=1
# accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=x)
# # Draw them
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
# #image = color.gray2rgb(image)
# for center_y, center_x, radius in zip(cy, cx, radii):
#     circy, circx = circle_perimeter(center_y, center_x, radius)
#     overlayimage[circy, circx] = 255
#
# print(radii)
# ax.imshow(overlayimage,cmap='gray')
# plt.show()







# python "F:\plt_img.py"
