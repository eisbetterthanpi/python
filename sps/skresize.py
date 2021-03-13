from skimage import io
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt

file="F:\sps python\jpged\M22 +IPTG YFP 4s.jpg"
save="F:\sps python\jpged\M22 +IPTG YFP 4.jpg"
image = plt.imread(file)

image=resize(image,(1024,1280))
# skimage.io.imsave(save,image)
io.imsave(save,image)



# import pil
# img=img.resize(1024,1280)
# from skimage.transform import rescale, resize
# img2=resize(img2,(1024,1280))
