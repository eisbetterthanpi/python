
# base cidr
# x,y="1.32.128.0","1.32.191.255"
# xi=x.split(".")
# xis=[format(int(i), "08b") for i in xi]
# # print(type(xis[1]))
# sx=''.join([format(int(i), "08b") for i in xi])
# ix=int(sx, 2)

def cidr(x,y): #ip range to cidr
    xi,yi=x.split("."),y.split(".")
    xis,yis=[format(int(i), "08b") for i in xi],[format(int(i), "08b") for i in yi]
    sx,sy=''.join(xis),''.join(yis)
    # ix,iy=bin(int(sx, 2)),bin(int(sx, 2))
    ix,iy=int(sx, 2),int(sy, 2)
    c=34-len(bin(iy-ix))
    # print(ix,iy,bin(iy-ix),34-len(bin(iy-ix)),bin(iy-ix)[2:].find('0'))
    if bin(iy-ix)[2:].find('0')!=-1: print("wwwwwwwwwwwwwwwww")
    return x+"/"+str(c)
# cidr(x,y)

# og get cidr
# import pandas as pd
# df = pd.read_csv('F:\study_club\cn_ip.csv', header=None)
# for index, row in df.iterrows():
#     # print(row[0],row[1])
#     print(cidr(row[0],row[1])+",")
# return cidr(iplr[0],iplr[1])

import pandas as pd
df = pd.read_csv('F:/study_club/au_ip.csv', header=None)
def getcidr(df): # get cidr list
    import numpy as np
    iplr = np.array(df.to_numpy())
    iprng = np.vectorize(cidr)(iplr[:,0],iplr[:,1])
    # iprng = np.core.defchararray.add(iprng,",")
    iprng = np.core.char.add(iprng,",")
    # print(iprng)
    # for i in iprng: print(i)
    iprange = pd.DataFrame(iprng)
    # iprange.to_csv('F:\study_club\chn_ip_range.csv', header=None)
    return iprange

df=getcidr(df)
# df = pd.read_csv('F:\study_club\chn_ip_range.csv', header=None)
def toblock(df):# cidr list to block
    # df_csv = pd.read_csv('F:\study_club\chn_ip_blk.csv', header=None)
    df_csv = pd.DataFrame()
    # print(df.shape, df.iterrows(), len(df.index))
    # df_csv=df.loc[rcount:rcount+256,0]
    # df_csv.iloc[rcount:rcount+256]=df.iloc[rcount:rcount+256]
    import math
    m=256
    n=math.ceil(len(df.index)/256)
    import numpy as np
    arr = np.array(df.to_numpy())[:,0]
    print(m,n,m*n, arr.size, df.index)
    arr = np.pad(arr, (0, m*n - arr.size), mode='constant', constant_values=np.nan).reshape(m,n, order='F')
    df_csv = pd.DataFrame(arr)
    # print(df_csv)
    # print(df.iloc[rcount:rcount+256])
    # df_csv.iloc[:,[rcount]] = df.iloc[rcount:rcount+256]
    df_csv.to_csv('F:/study_club/au_ip_blk.csv', header=None)
    return df_csv

df_csv=toblock(df)
