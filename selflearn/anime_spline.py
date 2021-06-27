import numpy as np
# anf=np.load('/content/drive/MyDrive/ans_100.npy')
ans=np.load('/content/drive/MyDrive/ans_convre100.npy')
# ans=np.load('/content/drive/MyDrive/ans.npy')

# anf=ans[:,:38,:]
anf=ans
print(anf.shape)
# print(anf)






# print(noise_vars)
import numpy as np

# size=Gs.input_shape[1:] #og [None, 512] [512]
# size=Gs.input_shape[-1]
size=512
step=0.1
r=1

al = np.random.uniform(-r,r,(1,size))
bl = np.random.uniform(-r,r,(1,size))
zs = np.full((1,size), 0.01) #rest is constant

zero = np.full(size, 0)
# ans = np.expand_dims(np.stack((zero ,zero), axis=1), axis=1)

# asl = np.stack((-np.full((1,size), 1), np.full((1,size), 1)), axis=2)
# asl = np.stack((-np.full(size, 1,dtype="float64"), np.full(size, 1,dtype="float64")), axis=1)
# asl=np.load('/content/drive/MyDrive/asl.npy')
asl=np.load('/content/drive/MyDrive/asl_convre100.npy')
# print(asl,asl.shape)

aslst = np.expand_dims(np.stack((zero ,zero), axis=1), axis=1)

rown=np.arange(size).reshape((size,1))

# def generate_ab_zs(chosen ,option, zs):
# def generate_ab_zs(chosen ,option, zs, ans=ans, r=1):
def generate_ab_zs(chosen ,option, zs, ans=ans, aslst=aslst, r=1):
    s = np.expand_dims(np.stack((chosen[0] ,option[0]), axis=1), axis=1)
    ans=np.concatenate((ans, s), axis=1)
    if ans.shape[1]<3:
        al = np.random.uniform(-r,r,(1,size))
        bl = np.random.uniform(-r,r,(1,size))
        pass
    else:
        # checkstep, assignstep, zspart = check_step()
        checkstep, assignstep, zspart = check_step(asl)
        asl[checkstep] = assignstep[checkstep]

        # s = np.expand_dims(np.stack((chosen[0] ,option[0]), axis=1), axis=1)
        s = np.expand_dims(asl, axis=1)
        aslst=np.concatenate((aslst, s), axis=1)
        # np.concatenate((aslst, s), axis=1,out=aslst)
        # aslst.append(s)

        al, bl = gen_absl(asl)
        # zs=(al+bl)/2
        zs=np.array([zspart])
        # print(np.append(rown,asl,axis=1)[np.any(asl!=[-1,1],axis=1)])
    return al, bl, zs, ans, aslst
# al, bl, zs, ans = generate_ab_zs(al, bl, zs, ans)

import numpy as np
def get_one(p=0):
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
  # xs = np.linspace(-1, 1, 100)
  xs = np.linspace(x[0], x[-1], 100)
  return xs, s(xs), s

def get_loss_spline(t, data, s):
  # loss=sum((s(t)-data)**2)/len(t) #msd
  loss=(sum((s(t)-data)**2)/len(t))**(1/2) #rmsd
  return loss

def plt_one(p=0):
  x,y = get_one(p)
  xs, ys, s = spline_one(x,y)
  g=s.derivative()

  plt.scatter(x,y,1)
  plt.plot(xs, ys, label='spline')
  plt.plot(xs, g(xs), label='grad')
  # plt.plot(g(xs), xs, label='grad')
  plt.show()
# plt_one(p)

def check_step(asl):
    dmax=np.array([])
    dmin=np.array([])
    smax=np.array([])
    smin=np.array([])
    gap=np.array([])
    l=[]
    assignstep=np.array([[0,0]])
    zspart=np.array([])

    for p in range(512):
        x,y = get_one(p)
        a=np.array([asl[p][0]<x, x<asl[p][-1]])
        cran=a.all(axis=0)
        x,y = x[cran],y[cran]
        # print(p,len(x))
        if len(x)<10:
            s = lambda x:0
            g = lambda x:0
            xs = np.linspace(asl[p][0], asl[p][-1], 100)
            ys = np.full(100, 0,dtype="float64") # one = np.full((1,size), 1)
        else:
            xs, ys, s = spline_one(x,y)
            g=s.derivative()
        # xs, ys, s = spline_one(x,y)
        # g=s.derivative()
        loss=get_loss_spline(x, y, s)
        l.append(loss)
        # abstep=np.array([assign_step(g, xs)])
        abstep=np.array([assign_step(g, xs)[0]])
        zpart=np.array([assign_step(g, xs)[1]])

        dmax=np.append(dmax,y.max()-y.min())
        dmin=np.append(dmin,ys.max()-ys.min())
        smax=np.append(smax,np.abs((y.max()-y.min())/ys.max()-ys.min()))
        smin=np.append(smin,abs(y-s(x)).max())
        gap=np.append(gap,np.diff(x).max())
        assignstep=np.append(assignstep,abstep,axis=0)
        zspart=np.append(zspart,zpart)

    # np.mean(smax)
    t1=dmax.min()+0.2*(np.mean(dmax)-dmax.min())
    t2=dmin.min()+0.2*(np.mean(dmin)-dmin.min())
    t3=smax.min()+0.2*(np.mean(smax)-smax.min())
    t4=smin.min()+0.2*(np.mean(smin)-smin.min())

    # a=np.array([dmax<0.3, dmin<1, smax<3 , smin<1, gap<0.09])
    # a=np.array([dmax<t1, dmin<t2, smax<t3 , smin<t4, gap<0.09])
    a=np.array([smax<3 , dmax<0.3, gap<0.09])
    # return a.all(axis=0), assignstep[1:]
    return a.all(axis=0), assignstep[1:], zspart

def assign_step(g, xs):
    v=g(xs)>0
    if not np.any(v): return np.array([-1, 1]), 0
    a=xs[0]+(xs[v][0]-xs[0])*0.5
    b=xs[-1]+(xs[v][-1]-xs[-1])*0.5
    zp=xs[np.argmax(g(xs))]
    return np.array([a, b]), zp

def gen_absl(asl):
    ab = np.random.rand(2,size)
    al,bl=ab*(asl[:,1]-asl[:,0])+asl[:,0] # (0,1) *(b-a)+a = (a,b)
    al=np.array([al])
    bl=np.array([bl])
    return al, bl





# p=
# for cut in range(100):
#     anf=ans[:,:cut,:]
#     anf=ans[:,:cut,:]

# num=np.arange(100)

anf=ans[:,50:100,:]
# asl = np.stack((-np.full(size, 1,dtype="float64"), np.full(size, 1,dtype="float64")), axis=1)
p=0 #12  24  32  37  46  55  76  77  92 119 132 145 189 199 289 290 313 331 372 384 428 453]
x,y = get_one(p)

a=np.array([asl[p][0]<x, x<asl[p][-1]])
cran=a.all(axis=0)
x,y = x[cran],y[cran]

plt.scatter(x,y,1)
plt.show()

if len(x)<10:
    s = lambda x:0
    g = lambda x:0
    # asl[p][0] asl[p][-1]
    xs = np.linspace(asl[p][0], asl[p][-1], 100)
    ys = np.full(100, 0,dtype="float64") # one = np.full((1,size), 1)
else:
    xs, ys, s = spline_one(x,y)
    g=s.derivative()

abstep, zsd=assign_step(g, xs)
# assignstep=np.array([[0,0]])
# zspart=np.array([])
# assignstep=np.append(assignstep,np.array([abstep]),axis=0)
# zspart=np.append(zspart,zsd)
# print("assignstep ",assignstep,zspart)

rown=np.arange(size) #.reshape((size,1))
checkstep, assignstep, zspart = check_step(asl)

print("rown ",rown[checkstep])
asl[checkstep] = assignstep[checkstep]
# print(".dtype ", asl.dtype, checkstep.dtype, assignstep.dtype)


plt.scatter(x,y,1)
plt.plot(xs, ys, label='spline')
plt.plot(xs, g(xs), label='grad')
plt.show()



# print(np.any(asl!=[-1,1],axis=1))
# print(np.append(rown,asl,axis=1))
# print(np.append(rown,asl,axis=1)[np.any(asl!=[-1,1],axis=1)])

ant=np.column_stack((rown, asl)) # ant=np.concatenate((rown, asl), axis=1)
# print(ant[np.any(asl!=[-1,1],axis=1)])
print(ant[p])

# al, bl = gen_absl(asl)







# ans=np.load('/content/drive/MyDrive/ans_convre100.npy')
# # ans=np.load('/content/drive/MyDrive/ans.npy')

# # anf=ans[:,:158,:]
# anf=ans
# # print(anf.shape,anf)

# aslst = np.expand_dims(np.stack((zero ,zero), axis=1), axis=1)
aslst=np.load('/content/drive/MyDrive/aslst.npy')

def make_aslst():
    aslst = np.expand_dims(np.stack((zero ,zero), axis=1), axis=1)
    for num in range(anf.shape[1]):
        al=np.array([anf[:,num,0]])
        bl=np.array([anf[:,num,1]])
        # print(al,al.shape)
        al, bl, zs, ans, aslst = generate_ab_zs(al, bl, zs, ans,aslst)

# print(aslst.shape,aslst)
# np.save('/content/drive/MyDrive/aslst.npy',aslst,allow_pickle=False)

# plot all
num=101
for lr in aslst:
  left=lr[:,0]
  right=lr[:,1]
  plt.plot(right-left)


# 29-324, 32-156, 37-370, 38, 459
# num=101
# lr=aslst[0]
# left=lr[:,0]
# right=lr[:,1]
# plt.plot(right-left)
# plt.plot(left)
# plt.plot(right)
#
# plt.show()
# # print(lr)
