'''
#og attempt
n=int(input("number:"))
s=[]
p=[2]
x=int(2)
a=int(0)
while p[-1]<=(n**(0.5)):
    while a<=len(p)-1:
        if x%p[-1]!=0:
            p.append(x)
            if n%p[-1]==0:
                s.append(p[-1])
                n=n//[p-1]
            x+=1
        a+=1
s.append(n)
if len(s)==1: print("prime")
else: print (s)
'''

#good for smaller numbers
def pls(n):
    #n=int(19483738)
    s=[]
    p=[2]#,3,5,7,11]
    x=int(2)
    while p[-1]<=(n**(0.5)):
        while n%p[-1]==0:
            s.append(p[-1])
            #s+=p[-1]
            n=n//p[-1]
        for a in p:
            if x%a==0:
                x+=1
        p+=[x]
        x+=1
    
    s.append(n)
    if len(s)==1: return "prime"
    else: return s
    #print(p)

if __name__ == '__main__':
    
    import time
    
    #print(p(input("number:")))
    t=int(738)#18977981
    
    start=time.time()
    print(pls(t))
    print('pls',time.time()-start)
    
