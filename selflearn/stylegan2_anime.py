


# https://github.com/NVlabs/stylegan2
# git clone https://github.com/NVlabs/stylegan2.git
# url = 'https://drive.google.com/open?id=1WNQELgHnaqMTq3TlrnDaVkyrAH8Zrjez'


import tensorflow as tf

print('Tensorflow version: {}'.format(tf.__version__) )

# Use '%' prefix in colab or run this in command line
# %cd /content/stylegan2

import pretrained_networks

folder='F:/selflearn/stylegan2-master/'

# It returns 3 networks, we will be mainly using Gs
# _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
# _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
# Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
_G, _D, Gs = pretrained_networks.load_networks(folder+'models/pre-trained_Anime(AaronGokaslan).pkl')


import numpy as np

def generate_zs_from_seeds(seeds):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        zs.append(z)
    return zs



import dnnlib
import dnnlib.tflib as tflib
import PIL.Image
from tqdm import tqdm

# Get tf noise variables, for the stochastic variation
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

# Trunctation psi value needed for the truncation trick
def generate_images(zs, truncation_psi):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if not isinstance(truncation_psi, list):
        truncation_psi = [truncation_psi] * len(zs)

    imgs = []
    for z_idx, z in tqdm(enumerate(zs)):
        Gs_kwargs.truncation_psi = truncation_psi[z_idx]
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        imgs.append(PIL.Image.fromarray(images[0], 'RGB'))

    # Return array of PIL.Image
    return imgs

def generate_images_from_seeds(seeds, truncation_psi):
    return generate_images(generate_zs_from_seeds(seeds), truncation_psi)










import numpy as np

def generate_zs_from_seeds(seeds):
    zs = []
    rnd = np.random.RandomState(2)
    # print(rnd) #RandomState(MT19937)
    # zn = rnd.randn(508)
    # zn=np.zeros(508)
    zn = np.full(508, 0.01) #rest is constant
    print(zn[0])
    # for seed_idx, seed in enumerate(seeds):
    for i in seeds:
      for j in seeds:
        for k in seeds:
          for l in seeds:

            a=np.array([i,j,k,l])
            z=np.append(a,zn)
            z=np.array([z])
            zs.append(z)
    return zs



import dnnlib
import dnnlib.tflib as tflib
import PIL.Image
from tqdm import tqdm

# Get tf noise variables, for the stochastic variation
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

# Trunctation psi value needed for the truncation trick
def generate_images(zs, truncation_psi):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if not isinstance(truncation_psi, list):
        truncation_psi = [truncation_psi] * len(zs)

    imgs = []
    for z_idx, z in tqdm(enumerate(zs)):
        Gs_kwargs.truncation_psi = truncation_psi[z_idx]
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        imgs.append(PIL.Image.fromarray(images[0], 'RGB'))

    # Return array of PIL.Image
    return imgs

def generate_images_from_seeds(seeds, truncation_psi):
    return generate_images(generate_zs_from_seeds(seeds), truncation_psi)



import dnnlib
import dnnlib.tflib as tflib
import PIL.Image
from tqdm import tqdm

# Get tf noise variables, for the stochastic variation
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

# Trunctation psi value needed for the truncation trick
def generate_images(zs, truncation_psi):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if not isinstance(truncation_psi, list):
        truncation_psi = [truncation_psi] * len(zs)

    imgs = []
    for z_idx, z in tqdm(enumerate(zs)):
        Gs_kwargs.truncation_psi = truncation_psi[z_idx]
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        imgs.append(PIL.Image.fromarray(images[0], 'RGB'))

    # Return array of PIL.Image
    return imgs

def generate_images_from_seeds(seeds, truncation_psi):
    return generate_images(generate_zs_from_seeds(seeds), truncation_psi)


from math import ceil

def createImageGrid(images, scale=0.25, rows=9):
   w,h = images[0].size
   w = int(w*scale)
   h = int(h*scale)
   height = rows*h
   cols = ceil(len(images) / rows)
   width = cols*w
   canvas = PIL.Image.new('RGBA', (width,height), 'white')
   for i,img in enumerate(images):
     img = img.resize((w,h), PIL.Image.ANTIALIAS)
     canvas.paste(img, (w*(i % cols), h*(i // cols)))
   return canvas





def generate_ab_zs(chosen ,option, zs):
    size=Gs.input_shape[1:]
    stdir = np.where(chosen > 0, 1, -1)
    zs=zs+ step*stdir
    # al = np.full(size, step) + np.random.rand(size) #uniform
    # bl = np.full(size, -step) + np.random.rand(size)
    randlr = np.random.choice((-1, 1), size=size)
    al = np.random.normal(loc=randlr*step, scale=1.0, size=size)
    bl = np.random.normal(loc=-randlr*step, scale=1.0, size=size)
    return np.array([al]), np.array([bl]), zs




# import time
size=Gs.input_shape[1:]
step=0.1
# al = np.full(size, step) + np.random.rand(size) #uniform
# bl = np.full(size, -step) + np.random.rand(size)
al = np.array([np.random.normal(loc=step, scale=1.0, size=size)])
bl = np.array([np.random.normal(loc=-step, scale=1.0, size=size)])
zs = np.full(size, 0.01) #rest is constant
# ak=np.append(al,bl,axis=0)
# while True:
if True:
  f=input()
  imgs = generate_images([al,bl], 0.5)
  display(createImageGrid(imgs, scale=0.5, rows=1))
  print(5*"\n")
  tap=input("j l")
  # tap="a"
  if tap in ['a','j']:
    al, bl, zs = generate_ab_zs(al, bl, zs)
  elif tap in ['d','l']:
    al, bl, zs = generate_ab_zs(bl, al, zs)
  else:
    print('invalid input')
    # continue
  



# generate 9 random seeds
# seeds = np.random.randint(10000000, size=9) #og
seeds = np.linspace(-1, 1, num=3)
# seeds=np.array([90])
# print(seeds)

zs = generate_zs_from_seeds(seeds)
# print(zs)
# print(type(zs))
# print(zs[0])
# print(type(zs[0]))
imgs = generate_images(zs, 0.5)
















# # generate 9 random seeds
# seeds = np.random.randint(10000000, size=9)
# print(seeds)
#
# zs = generate_zs_from_seeds(seeds)
# imgs = generate_images(zs, 0.5)
#
#
# img[0]
