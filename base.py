

def dec2base(n):
    base=3
    suh='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # ans=''
    ans=suh[n%base]
    if n==0: return '0'
    while n//base>0:
        n=n//base
        # ans=str(n%base)+ans
        ans=suh[n%base]+ans
        # n=n//base
    return ans
# print(f(11))
def base2dec(ap='DEADBEEF'):
    base=16
    suh='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ap='DEADBEEF'
    ans=0
    for i,v in enumerate(reversed(ap)):
        ans+=suh.index(v)*base**(i)
    return ans








n=str(input("number:"))
b=int(input("base"))
m=int(input("to"))

l=int(len(n)-1)
t=int(0)

k=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

ans=[int(k.index(n[a]))*b**(l-a) for a,t in enumerate(n)]
answer=sum(ans)
print(answer)

#[print(a,t) for a,t in enumerate(n)]
s=[]
while answer>=m:
    s.insert(0,k[answer%m])
    answer//=m
s.insert(0,answer)
fin=''.join(str(j) for j in s)
print(fin)




'''
import math
a=math.log(answer,m)
print(a)
'''




'''
n=str(list(input("number:")))
b=int(input("base"))
m=int(input("to"))
s=[]
#[0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z]
k=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

t=j=int(0)
a=[list(n)]
e=len(a)-1
while e!=0:
    
    l=t=t+(a[len(a)-e-1]*b**e)
    e=e-1
    
while t//m>0:
    j=j+1
    t=t//m
while j>=0:
    s.append(l//m**j)
    l=l-l//m**j
    j=j-1
print(s,a)


#for a in n:
'''



