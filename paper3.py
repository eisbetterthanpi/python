# python "F:\paper3.py"
#http://www.radio-science.net/2017/10/dynamic-spectrum-spectrogram-using.html
#https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.signal.spectrogram.html
# https://dsp.stackexchange.com/questions/1593/improving-spectrogram-resolution-in-python

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import soundfile as sf
from scipy import fft
from matplotlib import cm

def specplus(f, t, Sxx): #3D
    # https://stackoverflow.com/questions/56788798/python-spectrogram-in-3d-like-matlabs-spectrogram-function
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    # from matplotlib.collections import PolyCollection
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # nt=np.arange(len(paths))#np.array([0,1,2,3]) # print(nt)
    x,y=np.meshgrid(t,f) # x,y=np.meshgrid(f,t)

    # ax.plot_surface(t[None, :], np.log10(f[:, None]), 10.0*np.log10(Sxx), cmap=cm.coolwarm) # plt.yscale('symlog') # ax.yaxis.set_scale('log')
    # ax.plot_surface(x,np.log10(y), 10*np.log10(Sxx), cmap=cm.coolwarm)
    ax.plot_surface(x,np.log10(y), np.log10(Sxx), cmap=cm.coolwarm)
    # ax.plot_surface(x,np.log10(y), Sxx, cmap=cm.coolwarm)
    # ax.plot_surface(np.log10(x),y, np.log10(Sxx), cmap=cm.coolwarm)

    # plt.rcParams['grid.linewidth'] = 0   # change linwidth
    # plt.rcParams['grid.color'] = "black"
    # ax.set_axis_off()
    ax.set_ylabel('Frequency [Hz]') # plt.xlabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]') # plt.ylabel('Time [sec]')
    # ax.xaxis.set_rotate_label(False)
    ax.set_zlabel('amp [dB]')
    plt.show()
    return

def spectrogram(f, t, Sxx):
    myfilter = (f>20) & (f<18000)
    f = f[myfilter]
    Sxx = Sxx[myfilter, ...]
    plt.yscale('symlog')
    # # plt.pcolormesh(t,f, Sxx)
    # plt.pcolormesh(t,f, 10.0*np.log10(Sxx), cmap=cm.coolwarm)
    # amp = 2 * np.sqrt(2)
    # plt.pcolormesh(t, f, np.abs(Sxx), vmin=0, vmax=amp, shading='gouraud')
    # plt.pcolormesh(t, f, 10.0*np.log10(np.abs(Sxx)))
    # plt.pcolormesh(t, f, np.log10(np.abs(Sxx)))
    plt.pcolormesh(t, f, np.log10(Sxx))
    # plt.pcolormesh(t/2, 2*f, np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return

def phase():
    return

def selectivy(folder='F:\piano 162\Piano in 162 Samples\Close\PedalOffClose'): #01-PedalOffMezzoPiano2Ambient
    # F:\piano 162\Piano in 162 Samples\Ambient\PedalOnAmbient
    # (01)-Pedal(Off/On)(vels)(1/2)(Close/Ambient)
    # path ='F:\piano 162\Piano in 162 Samples\Close\PedalOnClose\\49-PedalOnForte1Close.flac'
    # from os import listdir
    # from os.path import isfile, join
    # allfiles = [f for f in listdir(folder) if isfile(join(folder, f))] # list of '01-PedalOffForte1Close.flac'
    # note=[int(x[:1]) for x in allfiles] # 01
    # ltn=['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']

    vels=['Pianissimo','Piano','MezzoPiano','MezzoForte','Forte']
    paths=[]
    lfs=[]
    for x in range(1,89):#1,89
        # path=(str(x).zfill(2) +'-PedalOff'+ 'Forte' +'1Close')
        path=(str(x).zfill(2) +'-PedalOn'+ 'Forte' +'1Close')
        paths.append(path)
        lfs.append(27.5*2**((x-1)/12))
    # paths=[(str(x).zfill(2) +'-PedalOff'+ 'Forte' +'1Close') for x in range(0,88)]
    # lfs=[27.5*2**(x/12) for x in range(0,88)]

    # paths=['49'+'-PedalOff'+ x +'1Close' for x in vels] #create for note
    # lfs=[27.5*2**((49-1)/12) for x in vels] #27.5

    return paths, lfs

def stftit(y, sr):
    # f, t, Sxx = signal.spectrogram(y, sr,  nperseg=11025, noverlap=10000, window='hamming', nfft=None) #for time
    f, t, Sxx = signal.spectrogram(y, sr,  nperseg=44100, noverlap=40000, nfft=None,scaling='spectrum',return_onesided=True,mode='magnitude')
    # f, t, Sxx = signal.spectrogram(y, sr,  nperseg=88200, noverlap=80000,window='hamming', nfft=None)
    # f, t, Sxx = signal.spectrogram(y, sr,  nperseg=176400, noverlap=170000,window='hamming', nfft=None)

    # f, t, Sxx = signal.spectrogram(y, sr,  nperseg=44100, noverlap=40000,window='hamming', nfft=None)
    # https://stackoverflow.com/questions/51898882/scipy-spectrogram-vs-multiple-numpy-ffts
    # /https://medium.com/analytics-vidhya/breaking-down-confusions-over-fast-fourier-transform-fft-1561a029b1ab
    # f, t, Sxx = signal.spectrogram(y, sr,  nperseg=2**16, noverlap=2**15,window='hamming', nfft=None)
    # f, t, Sxx = signal.spectrogram(y, sr, nperseg=96000, noverlap=48000) #for freq
    # big seg for freq, small seg for time, bigger lap better
    # freq resolution = nperseg/sr
    # print(f[0],f[1]-f[0],f[-1]) #0,invseg , 24000
    # print(t[0],t[1]-t[0],t[-1])
    # print(f,len(f)) #list of 22051, 0 to 22050 diff of nperseg/sr
    # print(t,len(t)) #list of 772, 0.5 to 67.5... diff of 4100/44100 1-nperseg/noverlap
    # print(len(Sxx[:, 1]) / nperseg * (1 + noverlap) )
    return f, t, Sxx

def energy(f, t, Sxx,show=False):
    # energy = [np.log10(sum(x)) for x in zip(*Sxx)] #log easier to see, but popup log0 error
    energy = [sum(x) for x in zip(*Sxx)]
    # print(energy,t)
    if show == True:
        plt.plot(t,energy)
        # plt.plot(energy)
        plt.show()
    return energy

def crop(f, t, Sxx):
    e = energy(f, t, Sxx)#,True)
    # print(e)

    # plt.pcolormesh(t, freq, np.log10(data))
    # plt.show()

    # filter=np.where(e>0) ?
    # data = next((index for index,value in enumerate(e) if value != 0), None)
    # start = next((index for index,value in enumerate(e) if value >= 5), None) #if value != 0
    # end = next((index for index,value in enumerate(list(reversed(e))) if value > 0.01), None) #if value != 0
    start = next((index for index,value in enumerate(e) if value > 10**-7), None) #for no log
    end = next((index for index,value in enumerate(list(reversed(e))) if value > 10**-13), None)
    # end=np.argmin(smooth<=fly[m]+5)
    if end in (0,None): #gives empty array if end==0
        end=-int(len(e))
    t = t[start:-end]
    # Sxx = Sxx[...,start:end] ?
    Sxx = Sxx[:,start:-end]
    # e = energy(f, t, Sxx,True)
    return f, t, Sxx

def logsm(Sxx,sness=0.1):
# def logsm(Sxx,frame,sness=0.1):
    # now=np.polyfit(f,Sxx[:,frame], 5) #for approximating to polynomial
    # wha=np.poly1d(now)(f)
    # sness=0.1#01-20 1800-22050

    # data=np.log10(Sxx[:,frame])
    # data=np.log10(Sxx)
    data=np.array(Sxx)

    # x=1000
    # start=int(np.floor(x*10**-sness)) # start=int(np.floor(x*10**(1-sness)))
    # end=int(np.ceil(x*10**sness))+0 # end=int(np.ceil(x*10**(1+sness)))+0
    # print('s',start,end)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        wha=[np.mean(data[int(np.floor(x*10**-sness)):int(np.ceil(x*10**sness))+0]) for x in range(len(data))]
    # wha=[]
    # for x in range(len(data)):
    # # for x in range(1,30):
    #     start=int(np.floor(x*10**-sness))
    #     end=int(np.ceil(x*10**sness))+0
    #     # print('s',start,end)
    #     # print(np.mean(data[start:end]))
    #     wha.append(np.mean(data[start:end]))
    # print(wha)
    # plt.plot(wha)
    # plt.plot(f,wha)
    return wha

# def harmonics(f, t, Sxx,parts=24,show=False): #find harmonics from bright
def harmonics(f, t, Sxx,lf,parts=24,show=False): #find harmonics from bright
    import math
    sindex=[] #amp over noise
    findex=[] #index of partials
    first=0
    frame=3
    # peaki = bright(f, t, Sxx, show=False)
    # main=np.log10(Sxx[:,frame])
    # main=np.log10(np.mean(np.array(Sxx[:,:5]),axis=1)) # main=np.mean(np.array(Sxx[:,:5]),axis=1)
    main=np.log10(np.mean(np.array(Sxx[:,:10]),axis=1))
    wha=logsm(main,sness=0.1)
    con=main-wha

    floor = logsm(np.log10(np.mean(np.array(Sxx[:,-10:]),axis=1)),sness=0.1)
    # flev = np.mean(floor[1:])
    weit=[np.log(x)for x in reversed(range(len(floor)))]
    # weit=[np.log(x)for x in range(len(floor))]
    print('weit',weit)
    flev = np.average(a, weights=weit)

    print('flev',flev)

    # for sharpness
    tbon=0.85#0.5#2.5#2
    peaki=sright(f,main,wha,tbon,stepsize=10,show=False)
    peaki=np.array(peaki)
    lf=27.5#440
    # print(f[peaki]) #print(peaki)
    np.set_printoptions(suppress=True) # np.set_printoptions(precision=2,formatter={'float_kind':'{:f}'.format}) # print(np.round(f[peaki],1))
    print(f[peaki],1)
    # print(f[peaki]/lf)
    lhe=1.001
    # print(peaki/(lf*lhe))
    # print(np.diff([0]+f[peaki]))


    # for absolute val
    tboe=-8.1
    peake=sright(f,main,[0]*len(main),tboe,stepsize=10,show=False)
    peake=np.array(peake)
    # print(f[peake])
    # np.set_printoptions(suppress=True) # np.set_printoptions(precision=2,formatter={'float_kind':'{:f}'.format}) # print(np.round(f[peaki],1))
    print(f[peake],1)
    # print(np.diff([0]+f[peake]))


    pincept = f[peaki[np.in1d(peaki, peake, assume_unique=True)]]
    print('pincept',pincept)
    # everything = f[peaki[np.union1d(peaki, peake, assume_unique=True)]]


    # harmnum=np.round(peaki/(lf*lhe),0) #around rint print(type(harmnum[0])) still float
    # harmnum=harmnum.astype(int) #convert to list of int
    # # print(harmnum)
    #
    # # estimate ih first
    # # np.where(harmnum==hn)[0]
    # # pyon=np.split(harmnum, np.where(np.diff(harmnum) > 1)[0]+1)
    # pyon=np.split(harmnum, np.where(np.diff(harmnum) > 1)[0]+1)
    # print(pyon[:])
    # lh=[(peaki[x+1]/peaki[x])*((y)/(y+1)) for x,y in enumerate(pyon[0][:-1])]
    # print(lh,np.mean(lh))
    # # print(peaki[pyon[0]])

    # full list of harmonics
    # for hn in range(1,int(harmnum[-1])):
    # # for hn in range(1,harmnum[-1]):
    #     # np.where(harmnum==hn)[0]
    #     if len(np.where(harmnum==hn)[0])==0:
    #         .append(lf*)
    #     if len(np.where(harmnum==hn)[0])==1:
    #         .append()
    #     if len(np.where(harmnum==hn)[0])==1:


    # calculate ih
    # print(hn,np.where(harmnum==hn)[0])
    # ih=(peaki[x+1]/peaki[x])*(x/(x+1)) #4*lf*ih**3/3*lf*ih**2


    # cut=peaki[:9] #[0.1814059  0.24263039 0.30385488 ...
    # # step=[x/(y+1) for x,y in enumerate(cut)]
    # step=np.log10([y/(x+1) for x,y in enumerate(cut/lf)])
    # print('cut',cut/lf)
    # print('step',step)
    # sii=[10**(y/(x+1)) for x,y in enumerate(step)]
    # print('sii',sii,np.mean(sii))

    # remove=[6,7,9,13,15,18]
    # remd=np.delete(peaki,remove)

    # print(np.fft.fft(peaki))
    # plt.scatter([0.5]*len(peaki),f[peaki])
    ih=1.005

    # parts=np.log(peaki[-1]/lf)-0.001/np.log(ih) # parts=(np.log(peaki[-1]/lf)-np.log(parts))/np.log(ih) #peaki[-1]=lf*parts*ih**parts
    # parts=math.floor(peaki[-1]/lf)
    # print(peaki[-1],lf,parts)
    # while start < peaki[-1]:
    # for x in range(1,parts+1): #og new
    #     ih=1.005#np.mean(iindex)
    #     start=math.floor(lf*x)
    #     end=math.ceil(x*lf*(ih**x))+1
    #     # print(start,end)
    #
    #     # index of absolute max for each section
    #     filterindex=np.argmax(con[start:end]) #filterindex=np.argmax(main[start:end])
    #     fmax = f[start+filterindex] #443.0 same as print(f[maxindex])
    #     findex.append(start+filterindex)
    #     # pindex.append(fmax)
    #     plt.scatter([0.2,0.2,0.3],[f[start],f[end],fmax])
    #     # plt.scatter(0.4,f[res])
    #
    #     fmean=np.mean(con[start:end]) #round(np.mean(con[start:end]),4)
    #     # print('sness',con[start+filterindex],round(con[start+filterindex]-np.mean(con[start:end]),4))
    #     sness=round(con[start+filterindex]-fmean,4)
    #     sindex.append(sness) #sindex.append(round(con[start+filterindex]-fmean,4))


    # iharm=[(pindex[x+1]/(x+1))/(pindex[x]/x)for x in range(1,len(pindex)-1)] #inharmonicity
    # print('t',len(findex))
    iharm=[(findex[x+1]/(x+2))/(findex[x]/(x+1))for x in range(len(findex)-1)]
    # print(iharm,np.mean(iharm))
    # pindex=np.unique(pindex) #np remove dupes
    # print('p',pindex)
    # print('f',findex)
    if show==True:
        plt.xscale('symlog')
        # plt.plot(np.log10(main))
        plt.plot(f,main) # plt.plot(main)
        plt.plot(f,wha) # plt.plot(wha)
        plt.plot(f,con) # plt.plot(con)
        # plt.plot(np.log10(Sxx[:,frame]))
        plt.plot(f,floor)
        # plt.plot(flev)
        plt.hlines(flev,10**1,10*12)
        plt.scatter(f[peaki],[5]*len(peaki))
        plt.scatter(f[peake],[5.5]*len(peake))
        plt.show()
        plt.scatter([0.4]*len(peaki),f[peaki])
        plt.scatter([0.3]*len(peake),f[peake])
    return pincept#peaki#findex, iharm#pindex[1:]

def decay(pindex,t,Sxx,show=False):
    from scipy.signal import savgol_filter
    fly = smooth(Sxx)
    dkdur=[]
    for x,m in enumerate(pindex):

        # Sxx[x]

        # print(len(t))
        sness = int(len(t) *0.1)
        smooth = savgol_filter(np.log10(Sxx[m]), sness-1-sness%2, 1)
        # end=np.argmin(smooth<-10)
        end = next(x for x,y in enumerate(smooth) if y < fly[m])
        print(end,t[end],fly[m])

        dkdur.append(t[end])
        if show == True:
            plt.plot(t,np.log10(Sxx[m]),label='partial '+str(x+1))
            plt.plot(t,smooth,label='smooth '+str(x+1))
    if show == True:
        plt.ylabel('Amplitude [Db]')
        plt.xlabel('Time [s]')
        # plt.plot(np.log10(Sxx[pindex]))
        plt.legend(loc="upper right")
        plt.show()
    return dkdur

def wobble(pindex,f, t, Sxx):
    nsr=100
    # print(pindex)
    # print(Sxx[pindex[2]])
    # f, t, Sxx = signal.spectrogram(Sxx[pindex[1]], nsr,  nperseg=nsr, noverlap=int(round(nsr*0.7)),window='hamming', nfft=None)

    # freq resolution = nperseg/sr
    # print(f[0],f[1]-f[0],f[-1]) #0,invseg , 24000
    # print(t[0],t[1]-t[0],t[-1])

    sr=1/(t[1]-t[0])
    # print(sr)
    freq, time, data = signal.spectrogram(np.log(Sxx[pindex[1]]), sr,  nperseg=200, noverlap=190,window='hamming', nfft=None) #for freq
    # freq, time, data = signal.spectrogram(np.log(Sxx[pindex[0]]), sr,  nperseg=2, noverlap=1,window='hamming', nfft=None) #for time
    # print(freq, time, data)
    # print(len(time))
    # plt.pcolormesh(t, freq, np.log10(data))
    plt.pcolormesh(time, freq, np.log10(data))

    # frame=0
    # plt.plot(freq,np.log10(data[:,frame]))
    # plt.xscale('symlog')
    # # plt.ylabel('Amplitude [Db]')
    # # plt.xlabel('Frequency [Hz]')
    # # if show == True:
    # fmax = freq[(np.argmax(data[:,frame]))]
    # print(fmax)

    plt.show()
    return

def sright(f,main,fly,tbon=0,stepsize=10,show=False): #get index of over treshold, + stepsize
    con=main-fly #for sharpness against close
    # vcon=con[con>tbon] #list of value
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        icon=np.where(con > tbon)[0] #list of index
    # print(con,icon)
    # pyon=[y for x,y in enumerate(icon) if icon[x]-icon[x-1]==1] #not quite there
    # pyon=np.where(np.diff(icon) != stepsize)[0] #list of ending index of each group
    pyon=np.split(icon, np.where(np.diff(icon) > stepsize)[0]+1) #list of group of index
    res=[np.argmax([con[y] for y in x]) for x in pyon] #get max of each group

    # top=23
    # ind = np.argpartition(con, -top)[-top:] #fast get last 7
    # sharpi=ind[np.argsort(con[ind])[:]] #2704361 index of last 7 sorted

    # con=np.array([4,8,2,6,5,1,7,3])#52704361
    # ind = np.argpartition(con, -7)[-7:] #fast get last 7
    # print(ind[np.argsort(con[ind])[:]]) #2704361 index of last 7 sorted

    # np.argsort(-arr)[:3]
    # sorted( [(x,i) for (i,x) in enumerate(lst)], reverse=True )[:3]

    # peaki=[(pyon[x][res[x]]) for x in range(len(pyon))] #alternative
    peaki=[y[res[x]] for x,y in enumerate(pyon)] #answer list of index
    if show==True:
        plt.xscale('symlog')
        plt.plot(f,fly)
        # # plt.plot(icon,vcon)
        plt.plot(f,main)
        plt.plot(f,con)
        plt.show()
    return peaki

def bright(f, t, Sxx, show=False): #locate all bright lines
    frame = 3
    main=np.log10(Sxx[:,frame])
    # main=np.log10(np.mean(np.array(Sxx[:,:5]),axis=1)) # main=np.mean(np.array(Sxx[:,:5]),axis=1)
    wha=logsm(main,sness=0.1)
    # con=main-wha
    if show == True:
        plt.yticks(np.arange(np.floor(min(main)), np.ceil(max(main[1:]-wha[1:])), 1.0))
        plt.grid(b=True)
        plt.xscale('symlog')
        plt.plot(f,main)
        # plt.plot(f,fly)
        # plt.plot(f,floor)
        # plt.plot(f,main-floor)
        # plt.plot(f,main-fly)
        plt.plot(f,main-wha)
        # plt.hlines(fly[int(expfreq)],0,t[-1])#,label='floor'+str(x))
        plt.plot(f,wha)

        plt.ylabel('Amplitude [Db]')
        plt.xlabel('Time [s]')
        # plt.plot(np.log10(Sxx[pindex]))
        plt.legend(loc="upper right")
        plt.show()

    tbon=2
    peaki=sright(f,main,wha,tbon,stepsize=10)
    plt.scatter([0.4]*len(peaki),f[peaki])

    # # try:
    # tbon=0.5
    # peaki=sright(f,main,fly,tbon,stepsize=10)#,show=True)#show) #for sharpness against close
    # hbon=2#4.5
    # sharpi=sright(f,main,floor,hbon,stepsize=10)#,show=True) #for spike above floor
    # # top=23
    # # ind = np.argpartition(con, -top)[-top:] #fast get last 7
    # # sharpi=ind[np.argsort(con[ind])[:]] #2704361 index of last 7 sorted
    # # # print(con[peaki],sharpi)
    # intercept=[x for x in peaki if x in sharpi]
    #
    # # print(peaki,sharpi,intercept)
    # # print(f[peaki],f[sharpi],f[intercept])
    # plt.scatter([0.2]*len(peaki),f[peaki])
    # plt.scatter([0.3]*len(sharpi),f[sharpi])
    # plt.scatter([0.4]*len(intercept),f[intercept])
    # # print(f[intercept],f[intercept]/lf)
    # # plt.show()
    # # intercept=0

    return peaki#intercept

def getpartials(): #get num of partials across 88
    # paths, lfs = choosenotes()
    paths,lfs=selectivy()
    # folder='E:\\ableton sounds\piano 162\Piano in 162 Samples\Close\PedalOffClose'
    folder='F:\piano 162\Piano in 162 Samples\Close\PedalOnClose'
    numpartial=[]
    # inharm=[]
    # print(lfs)
    # inharm = np.array([])
    inharm=[]
    for x,y in enumerate(paths):
        print(y)
        path=folder+'\\'+paths[x]+'.flac'
        # pindex = partials(path,lfs[x],parts=165)
        # print(x,path,lfs[x])
        # print(x,paths[x],lfs[x])
        # pindex = partials(paths[x],lfs[x],parts=165)
        # print(pindex)
        # numpartial.append(len(pindex))

        y, sr = sf.read(path)
        # # print(y[:,0]) #left?
        f, t, Sxx=stftit(y[:,0], sr)
        # f, t, Sxx=crop(f, t, Sxx)
        # myfilter = (f>20) & (f<18000)
        # f = f[myfilter]
        # Sxx = Sxx[myfilter, ...]

        lf=lfs[x]
        peaki = harmonics(f, t, Sxx,lf,parts=24,show=False)
        # peaki = bright(f, t, Sxx, show=False)
        peaki = np.array(peaki)
        # inharm=[round(f[y]/lf/(x+1),4) for x,y in enumerate(peaki)]
        inharm.append([round(f[y]/lf/(x+1),4) for x,y in enumerate(peaki)])
        # print([round(f[y]/lf/(x+1),4) for x,y in enumerate(peaki)])

        # print([(findex[x+1]/(x+1))/(findex[x]/x)for x in range(1,len(findex)-1)])
        # np.append(inharm,[round(f[y]/lf/(x+1),4) for x,y in enumerate(peaki)])
        # inharm=inharm+[(peaki[x+1]/(x+1))/(peaki[x]/x)for x in range(len(peaki)+1)]
        # inharm+=[(findex[x+1]/(x+1))/(findex[x]/x)for x in range(len(findex)+1)]
        # iharm=[(findex[x+1]/(x+2))/(findex[x]/(x+1))for x in range(len(findex)-1)]

        # number of partials over 88
        # numpartial.append(len(peaki))
        numpartial.append(peaki)
        plt.scatter([x]*len(peaki),peaki)

        frame = 3
        main=np.log10(Sxx[:,frame])
        # main=np.log10(np.mean(np.array(Sxx[:,:5]),axis=1)) # main=np.mean(np.array(Sxx[:,:5]),axis=1)
        wha=logsm(main,sness=0.1)
        # main=np.log10(Sxx[:,frame])
        numpartial.append(main)

    x = np.array(range(0,88))
    # y = 27.5*2**((x-1)/12)
    # y = [27.5*2**((v)/12) for v in x]
    y = np.array([27.5*2**((v)/12) for v in x])
    ih=1.006
    for h in range(1,29):
        plt.plot(x, h*y*ih**h)
        # plt.plot(x, h*y*ih*h)
        # plt.plot(x, h*y)
    # plt.xscale('symlog')
    plt.ylim(0, 12000)

    # frame of 88 keys in 3d
    # numpartial = np.array(numpartial)
    # nt=np.arange(len(paths))
    # import warnings
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
        # specplus(nt,f, numpartial)

    # print('mean',[np.mean(x) for x in inharm])
    # # print('mean',np.mean(inharm))
    # print(inharm)

    # plt.plot(nt,numpartial)
    # plt.show()
    # plt.plot(numpartial) # plt.matshow(numpartial) # plt.imshow(numpartial) # plt.scatter(numpartial)
    plt.show()
    return #numpartial
#get dkdur across 88

if __name__ == '__main__':
    # python "F:\paper3.py"
    import time
    start=time.time()
    # folder =F:\piano 162\Piano in 162 Samples\Close\PedalOnClose'
    # path ='F:\piano 162\Piano in 162 Samples\Close\PedalOnClose\\49-PedalOnForte1Close.flac'
    path ='F:\piano 162\Piano in 162 Samples\Close\PedalOffClose\\49-PedalOffForte1Close.flac' #01 49 85
    # path ='F:\csound\cabbage c7.flac'
    y, sr = sf.read(path)
    f, t, Sxx=stftit(y[:,0], sr) #left?
    # f, t, Sxx=crop(f, t, Sxx)
    # myfilter = (f>20) & (f<18000)
    # f = f[myfilter]
    # Sxx = Sxx[myfilter, ...]
    # paths,lfs=selectivy()
    # energy(f, t, Sxx,True)
    lf=27.5#3520#27.5
    # lf=round(27.5*2**((1-1)/12),4)

    # s=[27.5, 54, 81, 107.5, 135, 162, 189.5, 216.5, 244.5, 272, 328, 357, 386.5, 416, 445]
    # s=np.array([27.5, 54, 81, 107.5, 135, 162, 189.5, 216.5, 244.5, 272, 328, 357, 386.5, 416, 445])
    # print(s/27.5)
    # [print(round(y/lf/(x+1),4)) for x,y in enumerate(s)]

    # peaki = bright(f, t, Sxx, show=False)#True) #intercept

    # peaki=harmonics(f, t, Sxx,lf,parts=8,show=False)
    peaki=harmonics(f, t, Sxx,lf,parts=8,show=True)
    # peaki,iharm=harmonics(f, t, Sxx,show=False)
    # np.mean(iharm)
    #
    # # print(f[peaki])
    # # print(f[peaki]/lf)
    # inharm=[round(f[y]/lf/(x+1),4) for x,y in enumerate(peaki)]
    # # remove=[6,7,9,13,15,18]
    # # remd=np.delete(peaki,remove)
    # # print(f[remd])
    # # print(f[remd]/lf)
    # # inharm=[round(f[y]/lf/(x+1),4) for x,y in enumerate(remd)]
    # print('inharm',inharm)

    # # # dkdur=decay(pindex,t,Sxx,True)
    # # # print(dkdur)
    # getpartials()
    # print(numpartials)
    # plt.plot(numpartials)
    # # # wobble(pindex,f, t, Sxx)
    print('time',time.time()-start)
    # spectrogram(f, t, Sxx)
    # fmax = spectrograph(f, t, Sxx,3) #-1
    # specplus(f, t, Sxx)
    # plt.show()
