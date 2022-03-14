

# https://scipy-lectures.org/advanced/image_processing/

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

# el = ndimage.generate_binary_structure(2, 1)

# a = np.zeros((7,7), dtype=np.int)
# a[1:6, 2:5] = 1
# #Erosion removes objects smaller than the structure
# # a2=ndimage.binary_erosion(a, structure=np.ones((5,5))).astype(a.dtype)
# a2=ndimage.binary_erosion(a, structure=np.ones((2,2))).astype(a.dtype)
# plt.imshow(a2, cmap=plt.cm.gray)
# plt.show()


# # np.random.seed(2)
# im = np.zeros((64, 64))
# num_pts=18
# x, y = (63*np.random.random((2, num_pts))).astype(int) # x, y = (63*np.random.random((2, 8))).astype(np.int)
# im[x, y] = np.arange(num_pts) #gradient
# print(np.arange(num_pts))
# plt.imshow(im, cmap=plt.cm.gray)
# plt.show()



# bigger_points = ndimage.grey_dilation(im, size=(5, 5), structure=np.ones((5, 5)))
# square = np.zeros((16, 16)) # all black
# square[4:-4, 4:-4] = 1 # white square at mid
# dist = ndimage.distance_transform_bf(square) #gradient greyer further from center
# dilate_dist = ndimage.grey_dilation(dist, size=(3, 3), structure=np.ones((3, 3))) # bigger square
# # plt.imshow(square, cmap=plt.cm.gray)
# # plt.imshow(dist, cmap=plt.cm.gray)
# # plt.imshow(dilate_dist, cmap=plt.cm.gray)
# plt.imshow(bigger_points, cmap=plt.cm.gray)
# plt.show()



# square = np.zeros((32, 32))
# square[10:-10, 10:-10] = 1
# x, y = (32*np.random.random((2, 20))).astype(np.int)
# square[x, y] = 1
# open_square = ndimage.binary_opening(square)
# eroded_square = ndimage.binary_erosion(square)
# reconstruction = ndimage.binary_propagation(eroded_square, mask=square)
# plt.imshow(square, cmap=plt.cm.gray)
# plt.imshow(open_square, cmap=plt.cm.gray)
# plt.imshow(eroded_square, cmap=plt.cm.gray)
# plt.imshow(reconstruction, cmap=plt.cm.gray)
# plt.show()



n = 10
l = 256
im = np.zeros((l, l))
# np.random.seed(1)
points = l*np.random.random((2, n**2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

mask = (im > im.mean()).astype(np.float)
mask += 0.1 * im
# img = mask + 0.2*np.random.randn(*mask.shape)
img=mask

# hist, bin_edges = np.histogram(img, bins=60)
# bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

binary_img = img > 0.5
plt.imshow(binary_img, cmap=plt.cm.gray)
plt.show()












# https://stackoverflow.com/questions/246525/how-can-i-draw-a-bezier-curve-using-pythons-pil
def make_bezier(xys): # xys  sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n-1)
    def bezier(ts): # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier

def pascal_row(n, memo={}): # return nth row of Pascal's Triangle
    if n in memo:
        return memo[n]
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0: # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    memo[n] = result
    return result





from PIL import Image
from PIL import ImageDraw

# if __name__ == '__main__':
#     im = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
#     draw = ImageDraw.Draw(im)
#     ts = [t/100.0 for t in range(101)]
#
#     xys = [(0, 50), (10, 90), (50, 100)] #start, curve to, end
#     bezier = make_bezier(xys)
#     points = bezier(ts)
#
#     xys = [(50, 100), (90, 100), (100, 50)]
#     bezier = make_bezier(xys)
#     points.extend(bezier(ts))
#
#     # print(points)
#     draw.polygon(points, fill = 'red')
#     # im.save('F:/nus/sp3172/out.png')
#     plt.imshow(im, cmap=plt.cm.gray)
#     plt.show()



# im = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
# draw = ImageDraw.Draw(im)
#
# # def treepts(points,x,s,y,t,z,u):
# def treepts(x,s,y,t,z,u,points=''):
#     ts = [t/100.0 for t in range(101)]
#     # xys = [(0, 50), (10, 90), (50, 100)] #start, curve to, end
#     xys = [x, s, y]
#     bezier = make_bezier(xys)
#     if points=='':
#         points = bezier(ts)
#     else:
#         points.extend(bezier(ts))
#     # points = bezier(ts)
#     # points.extend(bezier(ts))
#     xys = [y, t, z]
#     bezier = make_bezier(xys)
#     points.extend(bezier(ts))
#     xys = [z, u, x]
#     bezier = make_bezier(xys)
#     points.extend(bezier(ts))
#     # print(points)
#     return points
#
# [x,s,y,t,z,u]=(512*np.random.random((6, 2))).astype(int)
# points=treepts(x,s,y,t,z,u,'')
#
# [x,s,y,t,z,u]=(512*np.random.random((6, 2))).astype(int)
# points=treepts(x,s,y,t,z,u,points)
#
# draw.polygon(points, fill = 'black')
# # im.save('F:/nus/sp3172/out.png')
# plt.imshow(im, cmap=plt.cm.gray)
# plt.show()



# from PIL import Image
# w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
# img = Image.fromarray(data, 'RGB')
# img.save('F:/nus/sp3172/my.png')
# img.show()
