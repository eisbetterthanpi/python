import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

def drift(particle,plot=False):
  x = particle['x']
  y = particle['y']
  l=len(particle)

  x=np.array(x)
  y=np.array(y)
  z=np.arange(l)  ## z=np.linspace(0, 1, num=l)

  A = np.vstack([z,np.ones(l)]).T
  xm, xc = np.linalg.lstsq(A, x)[0]
  ym, yc = np.linalg.lstsq(A, y)[0]
  if plot:
    # plt.plot(z, xm*z+xc)
    # plt.plot(z, ym*z+yc)
    fig = plt.figure()
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x,y,z)
    ax.plot(xm*z+xc,ym*z+yc,z)
    plt.show()
  return xm/l,ym/l
  # return xm,ym

A=[[1,2],[1,2,3],[1,2,3,4],[1,3]]
B=[[55,225,239],[2,50,70,104,108],[17,114,241],[3,34,73,181,241]]
folder="F:\sps python\\brownian\\results\\"

def getdsm(select,i):
    s={"A":A,"B":B}[select]
    ds=np.array([[0,0]])
    for j in s[i]:
        file='%s_%s_%s.csv' % (select,i+1,j)
        # print("file",file)
        # df_xy=glob.glob('results/'+file)[0] #colab
        df_xy=glob.glob(folder+file)[0] #windows
        particle = pd.read_csv(df_xy, usecols = ['x', 'y'])
        xdrift,ydrift=drift(particle)
        ds=np.append(ds,[[xdrift,ydrift]],0)
    ds=ds[1:]
    # print("ds",ds)
    # print(np.mean(ds,axis=0))
    return np.mean(ds,axis=0)

# dsmean=np.array([0.01860819, 0.00959338]) #all
def run(select,i):
    # select="A"
    s={"A":A,"B":B}[select]
    # for i in range(4):
    # i=3
    for j in s[i]:
        file='%s_%s_%s.csv' % (select,i+1,j)
        print(file)
        df_xy=glob.glob(folder+file)[0]
        particle = pd.read_csv(df_xy, usecols = ['x', 'y'])
        xdrift,ydrift=drift(particle,plot=True)
        # dsmean=[xdrift,ydrift] # use drift of 1
        dsmean=getdsm(select,i)
        l=len(particle)
        x = particle['x']
        y = particle['y']
        z=np.arange(l)
        particle['x']=x-dsmean[0]*z*l
        particle['y']=y-dsmean[1]*z*l
        xdrift,ydrift=drift(particle,plot=True)
    return particle
particle=run("B",3)


# df_xy=glob.glob(folder+'B_1_225.csv')[0]
# particle_A_1_xy = pd.read_csv(df_xy, usecols = ['x', 'y'])

def undrift(ogparticle):
  particle=ogparticle.copy()
  xdrift,ydrift=drift(particle,plot=True)
  l=len(particle)
  x = particle['x']
  y = particle['y']
  z=np.arange(l)
  particle['x']=x-dsmean[0]*z*l
  particle['y']=y-dsmean[1]*z*l
  xdrift,ydrift=drift(particle,plot=True)
  return particle


# particle=undrift(particle_A_1_xy)
# xdrift,ydrift=drift(particle,plot=True)





# df_xy=glob.glob(folder+'B_2_2.csv')[0]
# particle = pd.read_csv(df_xy, usecols = ['x', 'y'])
# # particle = read(glob.glob('results/B_1_225.csv'))
#
# xdrift,ydrift=drift(particle,plot=True)
# # print(xdrift,ydrift)
#
# [-1.14567215e-05 -4.82844549e-04] #A all
# # dsmean=np.array([-0.00479452,  0.0122292]) #B55
# # dsmean=np.array([0.00462271, 0.01662714]) #new first 3
# dsmean=np.array([0.01860819, 0.00959338]) #all
# # dsmean=np.array([xdrift,ydrift])
# l=len(particle)
# x = particle['x']
# y = particle['y']
# z=np.arange(l)
# print(x,x-dsmean[0]*z)
# particle['x']=x-dsmean[0]*z*l
# particle['y']=y-dsmean[1]*z*l
# # print(x,particle['x'])
# xdrift,ydrift=drift(particle,plot=True)

# print(xdrift,ydrift)






# particle = read(glob.glob('results/B_1_225.csv'))
# drift=np.array([[0,0]])
# for i in range(4):
#   for j in B[i]:
#     file='results/B_%s_%s.csv' % (i+1,j)
#     particle = read(glob.glob(file))
#     x = particle['x']
#     y = particle['y']
#     l=len(particle)
#     xdrift=(x[l-1]-x[0])/l
#     ydrift=(y[l-1]-y[0])/l
#     drift=np.append(drift,[[xdrift,ydrift]],0)
# print(drift)



# Z, B, C = 400, 300, 20
# Zs = []
# Bs = []
# for i in range(Z):
#     X, y, = make_regression(n_samples=B, n_features=C, random_state=i)
#     Zs.append(X)
#     Bs.append(y)
# Zs = np.array(Zs)
# Bs = np.array(Bs)
# result = np.empty((Z, C))
# for z in range(Z):
#     result[z] = np.linalg.lstsq(Zs[z], Bs[z])[0]

# https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
# x=np.array(x)
# y=np.array(y)
# x=np.reshape(np.array(x), (l,1))
# y=np.reshape(np.array(y), (l,1))
# z=np.arange(l)  ## z=np.linspace(0, 1, num=l)
# z=np.reshape(z, (l,1))
# print(x,z)

# data = np.concatenate((x,y,z),axis=1)
# datamean = data.mean(axis=0)

# xz = np.concatenate((x,z),axis=1)
# print(xz,z)

# A = np.vstack([x,np.ones(l)]).T
# print(A,z)
# m, c = np.linalg.lstsq(A, z)[0]
# print(m,c)

# m,c=np.linalg.lstsq(xz,z)
# u, s, vh = np.linalg.svd
# Do an SVD on the mean-centered data.
# uu, dd, vv = np.linalg.svd(data - datamean)
# for x in vv:
#     print(x)
# linepts = vv[0] * np.mgrid[-17:27:2j][:, np.newaxis]
# # shift by the mean to get the line in the right place
# linepts += datamean
# ax = m3d.Axes3D(plt.figure())
# ax.scatter3D(*data.T)
# ax.plot3D(*linepts.T)
# plt.show()

# dist(particle_A_2_xy)
# print(particle_A_2_xy)
