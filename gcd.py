def gcd(x,y):
    x,y=abs(x),abs(y)
    if y==0:
        return x
    return gcd(y,x%y)

x,y=input("x,y: ").split(",")
x,y=int(x),int(y)
print(gcd(x,y))

if __name__=="__main__":
    x,y=input("x,y: ").split(",")
    x,y=int(x),int(y)
    print(gcd(x,y))
