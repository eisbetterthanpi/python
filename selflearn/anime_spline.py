import numpy as np
# np.save('/content/gdrive/My Drive/ans.npy',ans,allow_pickle=False)
# https://colab.research.google.com/drive/1k3ePWc6pSeaIKicpLodOo9qzMhFuRGKv?authuser=2#scrollTo=B8J941HhhZIG

# anf=np.load('F:/selflearn/stylegan2-master/ans.npy')
# anf=np.load('F:/selflearn/stylegan2-master/ans_100.npy')
ans=np.load('F:/selflearn/stylegan2-master/ans_100.npy')
# print(anf.shape)
# anf=anf[:,:4,:]
# print(anf)



import numpy as np

def get_one(p=0, anf=ans):
  s0=anf[p][:,0]
  s1=anf[p][:,1]
  size=len(anf[p])
  one = np.full(size, 0.01) # one = np.full((1,size), 1)
  ss0=np.stack((s0, one), axis=1)
  ss1=np.stack((s1, -one), axis=1)
  sa=np.append(ss0,ss1,axis=0)
  sone = np.array(sa[np.argsort(sa[:,0])])
  cone=np.cumsum(sone[:,1])
  return sone[:,0],cone

# https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np

def spline_one(x,y):
    s = UnivariateSpline(x, y, s=1)
    xs = np.linspace(-1, 1, 100)
    return xs, s(xs), s

def get_loss_spline(t, data, s):
    # loss=sum((s(t)-data)**2)/len(t) #msd
    loss=(sum((s(t)-data)**2)/len(t))**(1/2) #rmsd
    return loss

# def plt_one(p=0):
def plt_one(p=0, anf=ans):
  x,y = get_one(p,anf)
  xs, ys, s = spline_one(x,y)
  g=s.derivative()

  plt.scatter(x,y,1)
  plt.plot(xs, ys, label='spline')
  plt.plot(xs, g(xs), label='grad')
  plt.show()
# plt_one(p=5)
# plt_one(anf=ans)

# def plt_more(p=0):
def plt_more_spline(p=0):
  x,y = get_one(p)
  # xs, ys, s = spline_one(x,y)
  s = UnivariateSpline(x, y, s=0.1)
  xs = np.linspace(-1, 1, 100)
  ys = s(xs)
  g=s.derivative()
  ax[p].scatter(x,y,1)
  ax[p].plot(xs, ys, label='after fitting')
  ax[p].plot(xs, g(xs), label='after fitting')

# fig, ax = plt.subplots(nrows=8, ncols=5, figsize=(20, 20))
# ax = ax.flatten() #https://stackoverflow.com/questions/37967786/axes-from-plt-subplots-is-a-numpy-ndarray-object-and-has-no-attribute-plot
# for x in range(40):
#   # plt_more(x)
#   plt_more_spline(x)
# plt.show()


def get_dsmax(anf=ans):
    dmax=np.array([])
    dmin=np.array([])
    smax=np.array([])
    smin=np.array([])
    gap=np.array([])
    l=[]
    for p in range(512):
        x,y = get_one(p,anf)
        xs, ys, s = spline_one(x,y)
        loss=get_loss_spline(x, y, s)
        # print("loss",loss)
        l.append(loss)

        dmax=np.append(dmax,y.max()-y.min())
        dmin=np.append(dmin,ys.max()-ys.min())
        smax=np.append(smax,np.abs((y.max()-y.min())/ys.max()-ys.min()))
        smin=np.append(smin,abs(y-s(x)).max())
        gap=np.append(gap,np.diff(np.sort(x)).max())


    # for x in [dmax, dmin, smax, smin]:
    arr=np.arange(512)
    # plt.plot(l); plt.plot(dmax); plt.plot(dmin); plt.plot(smax); plt.plot(smin); plt.show()
    return dmax, dmin, smax, smin, gap

# anf=ans[:,:4,:]
# dmax, dmin, smax, smin, gap = get_dsmax(anf=ans)
# dmax, dmin, smax, smin, gap = get_dsmax(anf)
# dmax, dmin, smax, smin, gap = get_dsmax(anf=anf)

# tar=4
# anf=ans[:,:tar,:]
# print(anf[0])
# ga=anf[0][:,0]
# print(np.diff(np.sort(ga)).max())
# print(np.diff(ga).max())
# np.diff(x).max()

# dmax, dmin, smax, smin, gap = get_dsmax(anf=anf)
# print(gap)
# anf=anf[:,:4,:]
# for tar in range(2,40):
for tar in range(40,80):
    anf=ans[:,:tar,:]
    dmax, dmin, smax, smin, gap = get_dsmax(anf=anf)
    # print(gap) 100 0.11-0.39      300 0.032 0.018
    pnum=12

    np.mean(smax)
    t1=dmax.min()+0.2*(np.mean(dmax)-dmax.min())
    # t2=dmin.min()+0.2*(np.mean(dmin)-dmin.min())
    t2=dmin.max()-0.2*(dmin.max()-np.mean(dmin))
    t3=smax.min()+0.2*(np.mean(smax)-smax.min())
    t4=smin.min()+0.2*(np.mean(smin)-smin.min())

    # print((dmax<0.3)[:pnum])
    # print((dmin<3)[:pnum])
    # print((smax<3)[:pnum])
    # print((smin<3)[:pnum])
    # print((gap<3)[:pnum])
    # a=np.array([dmax<0.3, dmin<1, smax<3 , smin<1, gap<0.09])
    # a=np.array([dmax>t1, dmin>t2, smax<t3 , smin<t4, gap<0.09])
    a=np.array([dmax>0.2, dmin>0.1, smax<3 , smin<0.1, gap<0.09])
    # a=np.array([smax<3 , dmax<0.3])
    # print((a.all(axis=0))[:40])
    # print(a.all(axis=0))
    print(tar,np.where(a.all(axis=0))[0])
    

# dmax
# dmin
# smax
# smin


# dmax
# dmin
# smax
# smin
