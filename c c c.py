n=int(input("min:"))
m=int(input("max:"))
a=[]
while n<=m:
    s=[]
    while n!=1:
        if n%2==0:
            n=int(n//2)
            s.append(n)
        elif n%2==1:
            n=int(3*n+1)
            s.append(n)
    a.append(len(s))
    print(len(s))
    n=n+1
print(a)
