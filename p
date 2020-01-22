
def p(n):
    n,f,s=int(n),int(2),[]
    while f<=(n**(0.5)):
        if n%f==0:
            s.append(f)
            n=n//f
        else:f+=1
    s.append(n)
    if len(s)==1: return "prime"
    else: return s
    exit

if __name__ == '__main__':
    print(p(input("number:")))
    '''
    import timeit
    t = timeit.Timer("p(n)", 100)
    print(t.timeit())
    '''

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
import time
n,f,s=int(input("number:")),int(2),[]
start= time.time()
while f<=(n**(0.5)):
    if n%f==0:
        s.append(f)
        n=n//f
    else:f=f+1
s.append(n)
if len(s)==1: print("prime")
else: print(s)
end time.time()
print(end- start)
'''
'''
n,s=int(input("number:")),[]
for f in range(2,abs(n**(0.5))+1):
    while n%f==0:
        s.append(f)
        n=n//f
        continue
if len(s)==0: print("prime")
else: s.append(n),print(s)
'''
