def collatz(n):
    #n=int(input("number:"))
    n=int(n)
    s=[]
    while n!=1:
        if n%2==0:
            n=int(n//2)
            s.append(n)
        elif n%2==1:
            n=int(3*n+1)
            s.append(n)
    return s
if __name__ == '__main__':
    print(collatz(input("number:")))
