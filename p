
def p(n):
    n,f,s=int(n),int(2),[]
    while f<=(n**(0.5)):
        if n%f==0:
            s.append(f)
            #s+=[f]
            n=n//f
            #n//=f
        else:f+=1
    s.append(n)
    if len(s)==1: return "prime"
    else: return s
    exit


'''
#main
n,f,s=int(input("number:")),int(2),[]
while f<=(n**(0.5)):
    if n%f==0:
        s.append(f)
        n=n//f
    else:f=f+1
s.append(n)
if len(s)==1: print("prime")
else: print(s)
'''
'''
#shortened
n,s=int(input("number:")),[]
for f in range(2,int(abs(n**(0.5))+1)):
    while n%f==0:
        s+=[f]
        n//=f    #continue
if len(s)==0: print("prime")
else: s.append(n),print(s)
'''


#+2+4
# 2,3,5,7,11,13
#2+1+2+2+4+2+4
def p24(n):
    f,s=int(2),[]
    #n,f,s=int(input("number:")),int(2),[]
    while f<=4:
        #print('i',f)
        if n%f==0:
            s.append(f)
            n=n//f
        else:
            f+=1
        if f>(n**(0.5)):
            break
    
    while f<=(n**(0.5)):
        #print(f,2+(f+1)%3)
        if n%f==0:
            s.append(f)
            n=n//f
        else:
            #f+=1
            
            f+=int(2+(f+1)%3)
    s.append(n)
    if len(s)==1: return "prime"
    else: return s


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


def pls24(n):
    #n=int(19483738)
    s=[]
    p=[2]#,3,5,7,11]
    f=int(2)
    
    while f<=4:
        #print('i',f)
        if n%f==0:
            s.append(f)
            n=n//f
        else:
            f+=1
        if f>(n**(0.5)):
            break
    while p[-1]<=(n**(0.5)):
        while n%p[-1]==0:
            s.append(p[-1])
            #s+=p[-1]
            n=n//p[-1]
        for a in p:
            if f%a==0:
                f+=int(2+(f+1)%3)
        p+=[f]
        f+=int(2+(f+1)%3)
        #print(f)
    s.append(n)
    if len(s)==1: return "prime"
    else: return s
    #print(p)




if __name__ == '__main__':
    
    import time
    
    
    #print(p(input("number:")))
    t=int(427495826018)#18977981
    
    start=time.time()
    print(p(t))
    print('p',time.time()-start)
    
    start=time.time()
    print(p24(t))
    print('p24',time.time()-start)
    
    start=time.time()
    print(pls(t))
    print('pls',time.time()-start)
    
    start=time.time()
    print(pls24(t))
    print('pls24',time.time()-start)
    
    '''
    from timeit import Timer
    t = Timer("p(184)", "from __main__ import p")
    print(t.timeit())
    '''




