
def test_leap_year(year):
    if divisible(year, 400):
        return 'Leap year!'
    elif divisible(year, 4) and not divisible(year, 100):
        return 'Leap year!'
    else:
        return 'Not a leap year!'
print(test_leap_year(1000))



i=-12
print(len("%i" % abs(i)))
return len("%i" % i)


def bsum(*a):
    s=sorted(a)
    return sum(map(lambda x:x**2 ,s[1:]))

def bsum(a,b,c):
    s=[a,b,c]
    # s=[a,b,c].sort()
    # print(m,type(m))
    # s.sort()
    print(s)
    return s[1]**2+s[2]**2

print(bsum(1,2,3))



n=6
a=n//2
# s=map(lambda x,y:x*y,range(1,a+1))
# # print(s)
# print(sum(map(lambda x,y:x*y,range(1,a+1))))
ans=0
# for y in range(0,a+1,2):
for y in range(0,n+1,2):
    f=1
    print(y)
    for x in range(1,y+1):
        # print(x)
        f*=x
    # print(f)
    ans+=f
print(ans)




a=2
b=3
print(a>0 and b>0)
print([x>0 for x in [a,b]])
if x<0 for x in [a, b]:
for x in [a, b]:
    # if x<0:
    if x<0 or type(x)!=int:
        print('po')
print('')
print([a,b]>0)


ip="00000011100000001111111111111111" #32 digits
def ip_format(ip):
    # Fill in your code here
    s=[int(ip[x*8:(x+1)*8],2) for x in range(4)]
    ans=str(s[0])
    for x in s[1:]:
        ans+='.'+str(x)
    return ans


a,b,c=1, -8, 15
a,b,c=3, 11, 9
print((-b+abs((b**2-4*a*c)**(1/2)))/(2*a))


def odd(num):
    s=[x if x in '13579' for x in str(num)]
    # s=[x in '13579' for x in str(num)]
    return sum(s)%2==1
print(odd(1299))



def format_sum(i):
    s=str(i).split('+')
    return sum([int(x) for x in s])
print(format_sum('1+2+3+4'))

list(map(int, time.split(":")))
import numpy as np
def triangle(*side):
    s=max([side.count(x) for x in side])
    # print(s)
    cut=[2*max(side)>=sum(side),s==3,s==2,s==1]
    print(cut.index(True))
    t=['Not a triangle','Equilateral','Isosceles','Scalene']
    # t=np.array(['Not a triangle','Equilateral','Isosceles','Scalene'])
    # print(max(side))
    print(t[cut.index(True)])
triangle(3, 3, 4)



# recurssion try
def son(n):
    if n==1: return 1
    return 2*n-1+son(n-1)
print(son(5))




def f(n):
    a=[1,1]
    for x in range(2,n+1):
        # a+=a[-1]+2*a[-2]+3*a[-3]
        a.append(a[-1]+a[-2])
    return a#[-1]


import math
def is_fib(n):
    # g=math.log(n)/math.log((5**(1/2)+1)/2)
    # print(g)
    # print((5*n**2+4)**(1/2),(5*n**2-4)**(1/2),(5*n**2+4)**(1/2)%1==0,(5*n**2-4)**(1/2)%1==0)
    return (5*n**2+4)**(1/2)%1==0 or (5*n**2-4)**(1/2)%1==0

# (((5**(1/2)+1)/2)**2-((5**(1/2)-1)/2)**2)/5**(1/2)
is_fib(-5)

def fib(x):
    # return math.ceil((((5**(1/2)+1)/2)**x-((5**(1/2)-1)/2)**x)/(5**(1/2)))
    # return ((5**(1/2)+1)**x-(5**(1/2)-1)**x)/(2**x*5**(1/2))
    return (5**(1/2)+1)**x/5**(1/2)
print([fib(x) for x in range(10)])

print(f(20))

a=[(5*n**2+4)**(1/2) for n in f(20)]
b=[(5*n**2-4)**(1/2) for n in f(20)]
print(a)
print(b)
(5*n**2-4)**(1/2)


def make_fare(stage1,stage2,start_fare,increment,block1,block2):
    def x(distance):
        if distance <= stage1:
            return start_fare
        elif distance <= stage2:
            return start_fare + (increment*ceil((distance - stage1) / block1))
        else:
            return taxi_fare(stage2) + (increment*ceil((distance - stage2) / block2))
    return x
#DO NOT REMOVE THIS LINE
comfort_fare = make_fare(1000, 10000, 3.0, 0.22, 400, 350)

higher order functions
def bas(a,b):
    def do(x):
        # print('x',x)
        return a*x+b
    return do
do=bas(2,3) # do is now a function that accepts x
print(do(3))




def occurrence(s1, s2):
    c=0
    x=0
    while x<=len(s1)-len(s2):
        print(x,s1[x:x+len(s2)])
        if s1[x:x+len(s2)]==s2:
            c+=1
            x+=len(s2)
        x+=1
    return c

print(occurrence('101010', '10'))



def perfect_number(n):
    c=0
    while n%2==0:
        # n%=2
        n=n/2
        c+=1
    return n==2**(c+1)-1
print(perfect_number(6))

def invert_number(num):
    n=str(num)
    # a=''
    # for x in range(0,len(n),-1):
    #     a+=n[x]
    a=n[::-1]
    return a
print(invert_number(1234))


import math
def reversed_numbers(low, high):
    di=math.floor(math.log10(low))+1
    a=10**(di//2+di%2-1)
    print('di,change,a',di,(di//2+di%2),a)
    s=(high-low)/10**(di-(di//2+di%2))
    print(high//10**(di-1)-low//10**(di-1),high%10-low%10-low//10**(di-1))
    if high%10-low%10-low//10**(di-1)>0: s+=high%10-low%10-low//10**(di-1)
    # low//10%10,high//10%10
    # if di>3:
    #     print(low//10%10,high//10%10)
    return math.floor(s)
print(reversed_numbers(1,99),reversed_numbers(150, 202),reversed_numbers(12, 21),reversed_numbers(1000, 3000))



def legendre(n):
    ta=[]
    for x in range(1,n):
        s=[x if ((x+1)%6==0 or (x-1)%6==0) for x in range(n**2, (n + 1)**2)]
        for x in s:
            if prime(x):
                ta.append(True)
                break
    return False not in ta


def legendre_n(n):
    a=0
    s=[]
    for x in range(n**2, (n + 1)**2):
        print(x)
        if prime(x):
            a+=1
            s.append(x)
    print(s)
    return a

def prime(n):
    if n in[2,3]: return True
    elif n<2: return False
    if n%2==0: return False
    x=3
    while x<=n**(1/2):
        if n%x==0: return False
        x+=2
    return True
print(prime(14))
print(legendre_n(3))


def factorial(n):
    s=map(lambda x,y:x*y,range(n+1))
    return s
print(factorial(5))
# lambda a, b: b*a(a, b-1)


def maclaurin(x, n):
    ans=0
    for g in range(n):
        ans+=x**(g)/factorial(g)
    return ans


def factorial(n):
    a=1
    for x in range(1,n+1):
        a*=x
    return a

print(maclaurin(2, 11))
for x in range(5):
    print(factorial(x))





def compose(f,g):
    return lambda x: f(g(x))
foo = lambda x: x+10
print(compose(foo, foo) (3))
three_x_plus_1 = lambda f: compose(add1,times3)(f)


def make_adder(i):
    def n(u):
        return i+u
    return n
add1 = make_adder(1)
print(make_adder(5)(10))


def is_same(x,y):
    return x == y
def make_verifier(key):
    def no(n):
        return is_same(n,key)
    return no
check_password = make_verifier(262010771)
print(check_password(262010771))

# works
def fold2(op, term, a, next, b, base):
    # print(type(op), type(term), type(a),type(next),type(b),type( base))
    if a > b:
        return base
    else:
        return op (term(a), fold2(op, term, next(a), next, b, base))

def geometric_series(a, r, n):
    # a*r**n
    def op(i,ii):
        return i+ii
    def term(ax):
        print(a,r,ax)
        return a*r**ax
    def next(t):
        return t+1
    return fold2(op, term, 1, next, n-1, a)

print(geometric_series(1/2, 1/2, 3))
print(geometric_series(1, 2, 4))



def num_combination(n, m):
    print(n,m)
    if m==1:
        return n
    return int(n*num_combination(n-1, m-1)/m)
print(num_combination(20, 4))

# s=('d',3)
# print(s)

bar = ("a", "b")#("a", "c") if is the same, python returns true but in cs1010s take as False
foo = ("a", "c")
bat = bar
bar = foo

print(bat,bar,foo)
print(bat is foo)   #

# print(type((3)))
# print(0/1 is 0)

# def square_odd_terms(t):
#     fn=lambda x: x**2 if x%2==1 else x
#     return (fn(t[0]), ) + map(fn, t[1:])


what?
def accumulate(combiner, base, term, a, next, b):
    if a>b:
        return base
    return combiner(term(a),accumulate(combiner, base, term, next(a), next, b))



def thrice(f):
    return lambda x: f(f(f(x)))
thrice(thrice(add1))(0) #9
thrice(thrice)(add1)(0) #27




def compose (f , g ):
    return lambda x : f ( g ( x ))

def thrice(f):
    return compose(compose (f , f ) , f )

identity = lambda x : x

def repeated (f , n ):
    if n == 0 :
        return identity
    else :
        return compose (f , repeated (f , n - 1 ))

def combine (f , op ,n ):
    result = f ( 0 )
    for i in range ( n ):
        result = op ( result , f ( i ))
    return result

sq = lambda x : x ** 2

n=9
def f(x):
    print('f')

# f=print('f')
# print(thrice(thrice(f()))(0))
# print(repeated(f(), n)(0))


# thrice(thrice(f))(0) # 9 times
# repeated(f, n)(0)

# def add1 ( x ): return x + 1
# thrice(thrice)(add1)(6)
# thrice(thrice)(identity)(compose)
# thrice(thrice)(sq)(1)
# thrice(thrice)(sq)(2)

# print(thrice)(add1)(6)

# print(thrice(thrice)(add1)(1))
# print(thrice(thrice)(identity)(compose))
# print(thrice(thrice)(sq)(1))
# print(thrice(thrice)(sq)(2))



mission 3
def smiley_sum(t):
    def f(x):
        return x**2
    def op(x, y):
        print(x,y)
        if y==1: return x+y
        return x+2*y
    n = t+1
    return combine(f, op, n)
print(smiley_sum(5))








def sums(nums=(6, 7, 11, 15, 3, 6, 5, 3), target=6):
    lookup = dict(((v, i) for i, v in enumerate(nums)))
    return next(( (i+1, lookup.get(target-v)+1)
            for i, v in enumerate(nums)
                if lookup.get(target-v, i) != i), None)

def sums(arr,targ):
    look_for = {}
    for n,x in enumerate(arr,1):
        try:
            return look_for[x], n
        except KeyError:
            look_for.setdefault(targ - x,n)

def sums(list,target):
    '''only for 2 sum'''
    k={value:index for index, value in enumerate(list)}
    ans=[[j,k[target-x]] for j,x in enumerate(list) if target-x in k]
    # print(k,ans)
    ans=[x for x in ans if x[0] != x[1]]
    print(ans)
    return ans[0]



def sums(list,target):
    if target in list:
        return [list.index(target)]
    ans,list_sorted=[],[]
    for i,v in enumerate(list):
        ans+=[[[i+1],v]] #list of indexes used, value of sum
        list_sorted+=[[i+1,v]]
    list_sorted=sorted(list_sorted,key=lambda l:l[1], reverse=True) #[index,value] sorted by value
    for y in list_sorted:
        for val in ans:
            if not y[0] in val[0]: #make sure index have not been used
                if val[1]+y[1]==target: #target achieved, done!
                    return sorted(val[0]+[y[0]])
                elif val[1]+y[1]<target: #add new possible sum to memory
                    ans+=[[val[0]+[y[0]],val[1]+y[1]]]

ls,t=[1,2,6,3,17,82,23,234],12   #[2,4,5]
print(sums(ls,t))




# m=((-1, 2, 4, 5, 6), (8, 9, 10, 12, 13), (12, 11, 10, 9, 8))
# s=[x for y in m for x in y] #this is the way
# # s=(x for y in m for x in y) #nope, generator object
# # s=[x for x in y for y in m] #error y not defined
# print(s)





# def operate_elevator(*t):
#     e1=(1,0,1)
#     e2=(2,0,1)
#     def f(e,t):
#         return (e[0],abs(e[1]-t[1])+abs(e[2]-t[2]),t[2])
#     for x in t:
#         print(x[0])
#         if x[0]==1:
#             e1=f(e1,x)
#         elif x[0]==2:
#             e2=f(e2,x)
#     return (e1,e2)


def operate_elevator(t1,t2):
    e1=[1,0,1]
    e2=[2,0,1]
    def f(e,t):
        print(e[2]-t[1],t[1]-t[2])
        return [e[0],e[1]+2*(abs(e[2]-t[1])+abs(t[1]-t[2])),t[2]]
    for x in (t1,t2):
        if x[0]==1:
            print(x[0])
            e1=f(e1,x)
            print(e1)
        elif x[0]==2:
            e2=f(e2,x)
    return (tuple(e1),tuple(e2))

# print(operate_elevator((2, 5, 8), (1, 9, 7)))
print(operate_elevator((1, 9, 7), (1, 3, 10))) #((1, 42, 10), (2, 0, 1))
# ((1, 20, 7), (2, 14, 8))

# def pascal(n):
#     if n==1: return 1
#     t=[1]
#     p=pascal(n-1)[-1]
#     print(p)
#     for x in range(n-1):
#         t+=p[x]+p[x+1]
#     t+=1
#     return pascal(n-1)+t
#
# print(pascal(2))


def pascal(n):
    if n==1: return ((1,),)
    s=[]
    for v in range(1,n):
        m=1
        for x in range(1,v):
            print(v,x)
            m=m*(n-x)/x
        s+=[int(m)]
    s+=[1]
    print(tuple(s))
    return (pascal(n-1))+(tuple(s),)

print(pascal(5))

# choose
n=5
s=[]
for v in range(1,n):
    m=1
    for x in range(1,v):
        print(v,x)
        m=m*(n-x)/x
    s+=[m]
s+=[1]
print(s)
#5 14641
#4*3/2   4*3*2/3*2



def shift_left(num, n):
    if n==0: return int(num)
    num=str(num)
    n=n%len(num)
    num=num[1:]+num[0]
    # num=num[-1]+num[:-1]
    return shift_left(num, n-1)

def shift_left_alt(num, n):
    num=str(num)
    n=n%len(num)
    num=num[n::]+num[0:n]
    # num=num[-n:]+num[:-n]
    return int(num)

# print(shift_left(12345,-1))
print(shift_left_alt(12345,3))


def mth_digit(n, num):
    print(len(str(num)))
    if len(str(num))==n: return num%10
    return int(mth_digit(n, num//10))
print(mth_digit(4, 12345))

def mth_digit(n, num):
    if n>len(str(num)): return None
    return int(str(num)[n-1])


def divisible_by_11(num):
    if num in(0,11):
        print("0",num)
        return True
    if 0<num and num<11:
        print("out",num)
        return False
    num=str(num)
    a=1
    b=1
    # print(num[0::2])
    for x in num[0::2]: a+=int(x)
    for x in num[1::2]: b+=int(x)
    print(a,b)
    return int(divisible_by_11(abs(a-b)))


def count_change(tt, ck):
    c=(1,5,10,20,50,100)
    d=(0,1,3,6,15,30)
    e=(1,2,4,10,57,315)
    n=tt
    s=tuple()
    ans=0
    for x in range(ck):
        ans+=(n//c[ck-x])*e[ck-x]
        print(x,(n//c[ck-x]),e[ck-x])
        s=s+(n//c[ck-x],)
        n=n%c[ck-x]
    print(n,s,ans)
    # return ans

count_change(120,5)


def copy_tree(t):
    ans=tuple()
    for x in t:
        if type(x)==tuple:
            # ans=ans+(copy_tree(x),)#(x,)
            ans=ans+(copy_tree(x),)+tuple()
        else:
            ans=ans+(x,)+tuple()
        # ans=ans+tuple(x)
    return ans+tuple()
s=(23,452,(23,4,2),23)
print(copy_tree(s))
print(copy_tree(s)==s,copy_tree(s) is s)



def test_prime(n):
    if n<= 1:
        return False
    if n== 2:
        return True
    if n%2 == 0:
        return False
    for i in range(3, round(n**(1/2))+1,2):
        if n%i == 0:
            return False
    return True
print(prime(34))



# consolidated midterms
a = 2
b = a**2 + a
c = b//a + b%a
if c%b > a:
    print("Higher")
elif c%b < a:
    print("Lower")
else:
    print("Same same!")
print(a,b,c,c%b)

print(6%3)
c,b=6,3
print(c%b)

def x(y):
    def y(z):
        x = lambda x: x**2
        y = lambda x: x+2
        def z(x):
            return x(5)
        return z(y)
    return y(x)
# print(x(lambda x: x+1))
print(x(87))

result = 8
for i in range(10,5,-1):
    result = result**2 % i
    if result%2 == 1:
        result = 2*result
print(result)


i, count = 3,0
while i != 1:
    if i%2==1:
        i = 3*i+1
    else:
        i = i//2
    print("if",count,i)
    if count > i:
        print(i,count)
        break
    count += 1
print(count)

def twice(f):
    return lambda x: f(f(x))
print(twice(twice)(twice(lambda x: x+3))(2)) #26    8
print(twice(twice)(twice)(lambda x: x+3)(2)) #50    16


def twice(f):
    return lambda x: f(f(x))
print(twice(twice)(twice(lambda x: x+1))(0)) # 8    2*2^2
print(twice(twice)(twice)(lambda x: x+1)(0)) # 16   2^2^2
print(twice(twice(twice))(lambda x: x+1)(0))

def thrice(f):
    return lambda x: f(f(f(x)))
print(thrice(thrice(lambda x: x+1))(0)) # *3 *3
print(thrice(thrice)(lambda x: x+1)(0)) # (*3)(*3)(*3)    *3^3

print(thrice(thrice)(thrice(lambda x: x+1))(0)) # 8    2*2^2
print(thrice(thrice)(thrice)(lambda x: x+1)(0)) # 16   2^2^2
print(thrice(thrice(thrice))(lambda x: x+1)(0))


s=(1,3,5,2)
f= lambda x,y:x+y
# print(f(s))
# print(map(f,s))
print([map(f,x) for x in s])
print(reduce(lambda x, y:x+y, s)) #reduce is not in python 3.7
print(reduce(f,s,0))
# print(sum(s))



def poly(c, x):
    # c[]
    # a=1
    def term(a):
        return c[a] * (x ** a)
    return sum(term, 1, 0, len(c))
    # return sum(term, x, next, 0)

def fill_row(b):
    at=False
    for x in b:
        if x.count(0)==1:
            at=True
            x[x.index(0)]=10-sum(x)
    return at

def fill_col(b):
    at=False
    for x in range(4):
        k=[g[x] for g in b]
        if k.count(0)==1:
            at=True
            b[k.index(0)][x]=10-sum(k)
    return at

def fill_section(b):
    at=False
    for t in range(4):
        x=[b[2*(t%2)+(i%2)][2*(t//2)+i//2] for i in range(4)]
        if x.count(0)==1:
            at=True
            b[2*(t%2)+(x.index(0)%2)][2*(t//2)+x.index(0)//2]=10-sum(x)
    return at



def rabbit(r):
    r=(0,)+r
    ck=[r[w+1]-r[w] for w in range(len(r)-1)]
    print(ck,any(i > 50 for i in ck))
    if any(i > 50 for i in ck): return -1
    ans=0
    c=0
    d=1
    # for x in r:
    for x in range(len(r)):
        # while
        # r[c:]
        print(x,r[x],r[c],r[x]-r[c])
        if r[x]-r[c]>50:
            c=x-1
            print("c",c)
            ans+=1
            continue
        # d+=1
        # c=x
    return ans+1

# print(rabbit((32, 46, 70, 85, 96, 123, 145)))
# print(rabbit((40, 70, 150, 160, 180)))
print(rabbit((50, 51, 101, 102, 152, 153, 203, 204, 254, 255, 305, 306, 356, 357, 407)))



def cache(st):
    ans=0
    a=[]
    for x in st:
        print(x,a,x in a)
        if x in a:
            a.remove(x)
            a+=[x]
            ans+=20
        else:
            a+=[x]
            ans+=100
            if len(a)>8:
                a.pop(0)
                print(a)
    return ans

print(cache((3, 51, 24, 12, 3, 7, 51, 8, 90, 10, 5, 24)))


def merge(t1, t2):
    c1,c2=0,0
    if not t1:
        return t2
    if not t2:
        return t1
    if t1[0]>t2[0]:
        t1,t2=t2,t1
    ans=tuple()
    while c1<len(t1) and c2<len(t2):
        if t1[c1]>t2[c2]:
            ans+=(t2[c2],)
            c2+=1
        else:
            ans+=(t1[c1],)
            c1+=1
    if c1>=len(t1):
        ans+=t2[c2:]
    else:
        ans+=t1[c1:]
    return ans
# g=(-3, 8, 65, 100, 207)
g=tuple()
# print(g==True)
# merge((-3, 8, 65, 100, 207), (-10, 20, 30, 40, 65, 80, 90))
merge(g, (-10, 20, 30, 40, 65, 80, 90))

def reverse_tuple(t):
    s=tuple()
    g=[x for x in t[::-1]]#s=s+(x,)
    # print([x for x in t[::-1]])
    [s=s+(x,) for x in g]
    return s
print(reverse_tuple((1, 2, 3, 4, 5, 6, 7, 8, 9)))

def repeat(s):
    al=[]
    p=[]
    for x in s:
        if not x in al:
            al+=[x]
        elif not x in p:
            p+=[x]
    return len(p)

def repeat(st):
    # s=[]
    c=0
    r=False
    ans=1
    for x in range(1,len(st)):
        if st[x-1]==st[x]:
            # s+=[]
            if r==False:
                c+=1
            r=True

            # if c>ans:ans=c
        else\
            r=False
            # c=0
    print(c)
    return c

repeat('hsSisSs')
repeat('hssisss')



n=5
s=""
for x in range(n):
    p=((x)%2)*" "
    # s+=p+(x//2)*"* "+"*\n"
    print(p+(x//2)*"* "+"*\n")

def triangle_iterative(n):
    s=""
    for x in range(1,n+1):
        # s+=n*"$"+"\n"
        s=s+n*"$"+"\n"
        # '$\n$$\n$$$\n$$$$\n$$$$$\n'
    print(s)
    # return s
triangle_iterative(5)

def makeTriangle(sign):
    def triangle(n):
        print(n)
        s=""
        for x in range(1,n+1):
            # s+=n*"$"+"\n"
            s=s+x*sign+"\n"
            # '$\n$$\n$$$\n$$$$\n$$$$$\n'
        print(s)
            # return triangle
        return s
    return triangle
    # def triangle_iterative(n):

makeTriangle('@')(5)


import math
def largest_square_pyramidal_num(n) :
    k=math.floor((3*n)**(1/3))
    if (2*k**3 + 3*k**2 + k)/6>n:
        return (2*(k-1)**3 + 3*(k-1)**2 + (k-1))/6
    # if (2*(k-1)**3 + 3*(k-1)**2 + (k-1))/6==n:
        # return n
    return (2*k**3 + 3*k**2 + k)/6
largest_square_pyramidal_num(9)

from math import *
def find_largest_num(test_func, n) :
    if n <= 0:
        return False
    elif test_prime(n) :
        return n
    else :
        return find_largest_num(test_func, n-1)

for x in range(-3,20):
    print(x,find_largest_num(test_prime, x))



# ugly remove dupe
def remove_extras(l):
    s=[]
    a=0
    z=len(l)
    # for c,x in enumerate(l):
    c=0
    # while c<=len(l):
    while c<z:
        x=l[c]
        print("cx",c,x)
        # c=c-a
        if x not in s:
            s+=[x]
        else:
            l.pop(c-a)
            # a+=1
            c-=1
            z-=1
        c+=1
        print("xs",x,s,l)
    print(l)
    return l

# str replace flatten
def count_occurrences(l, n):
    t=str(l)
    t=t.replace("[","")
    t=t.replace("]","")
    t=t.replace(" ","")
    t=t.split(",")
    s=t.count(str(n))
    # return str(l)
    return s


def flat(l):
    s=[]
    for i,x in enumerate(l):
        if type(x)==list: s+=flat(x)
        else: s+=[x]
    return s
print(flat(a))


def sort_age(l):
    s=[("m",0)]
    for x in l:
        print("x",x)
        # c=0
        for c in range(len(s)):
            print("c",s)
            print("sdv",x[1],s[c][1])
        # for c,p in enumerate(s):
            if x[1]>s[c][1]:
                c+=1
            else:
                s[c:c]=s[c]
                # s[c+1:c+1]=s[c]
                s[c:c]=l[c]
                print("e",s)
                break
            l.append(x)
    # return s
    print(s)
sort_age([("F", 18), ("M", 23), ("F", 19), ("M", 30)])


list1 = [1] * 4
list2 = [5, 5, 5]
# while not 0:
while 1:
    list1[0] += 1
    if list1[0] == 5:
         break
         list1[1] += 2
    list1[2] += 3
print(list1 , list2)    #[5, 1, 10, 1] [5, 5, 5]
print(list1 < list2)


def hanoi(num, st, ed, bu):
    def shift(num,st,ed):
        if num==1: return ((st,ed),)
        return shift(num-1,st,6-st-ed)+((st,ed),)+shift(num-1,6-st-ed,ed)
    return shift(num,st,ed)
print(hanoi(4, 1, 3, 2))

def transpose(m):
    return [[p[x] for p in m]for x in range(len(m[0]))]
def transpose(m):
    c=len(m)
    a=[]
    for x in range(len(m[0])):
        for p in range(c):
            a+=[m[p][x]]
        m.append(a)
        a=[]
    del m[0:c]
    return m#.pop(0:c)
m = [[ 1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(transpose(m))



def mode_score(s):
    p=[x[2] for x in s]
    t=0
    q=list(set(p))
    print(q)
    for x in q:
        print(x,p.count(x))
        if p.count(x)>t:
            a=[x]
            t=p.count(x)
        elif p.count(x)==t:
            a+=[x]
    return a


def top_k(s, k):
    a= sorted(s, key = lambda x: x[2],reverse=True)[:k]
    a+=[x for x in s if x[-1]==a[-1][-1] and not x in a]
    # return sorted(a, key=lambda x: (x[1], x[2]))
    return sorted(a, key=lambda x: (-x[2], x[0])) #reverse sort by 3rd element then sort 1st element



# hof with wierd code
def accumulate(op, init, seq):
    if not seq:
        return init
    else:
        return op(seq[0], accumulate(op, init, seq[1:]))

def accumulate_n(op, init, sequences):
    if (not sequences) or (not sequences[0]):
        return type(sequences)()
    else:
        return ( [accumulate(op, init, list(map(lambda x:x[0],sequences)))]
               + accumulate_n(op, init, list(map(lambda x:x[1:],sequences))) )

def letter_count(s):
    s=[x for y in s for x in y]
    c=list(set(s))
    return [[x,s.count(x)] for x in c]

def who_wins(m, n):
    '''terrorist'''
    c=len(n)%m
    d=len(n)//m
    print(len(n)%m)
    # return n[-(len(n)%m)]
# print(set(who_wins(3, ['val', 'hel', 'jam', 'jin', 'tze', 'eli', 'zha', 'lic'])))
print(who_wins(2, ['poo', 'ste', 'sim', 'nic', 'luo', 'ibr', 'sie', 'zhu']))



def translate(j,k,s):
    j,k=list(j),list(k)
    s=list(s)
    for x,y in enumerate(s):
        if y in j:
            s[x]=k[j.index(y)]
    return ''.join(s)
print(translate("dikn","lvei","My tutor IS kind"))

def calculate(i):
    if len(i)==1: return i[0]
    t=str(i)
    t=t.replace(")","")
    t=t.replace("(","")
    t=t.replace(" ","")
    t=t.replace(",","")
    t=t.replace("''","")
    t=list(t)
    t=[x for x in t if not x in [" ",",","'"]]
    c=0
    e=len(t)
    while c<e:
        if t[c] in ['+', '-', '*', '/']:
            d=t.pop(c-1)
            t.insert(c,d)
        c+=1
    for x in range(int((len(t)-1)/2)):
        print("z",t,''.join(t[:3]))
        exec("z=" + ''.join(t[:3]), globals())
        t=[str(z)]+t[3:]
    return z
print(calculate((1, 2, '+', 3, '*')))

s = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print(type(s))
f=[x for x in s]
# [s[x]=s[x]+1 for x in f]
for x in f:
    s[x]+=1
print(f,s)


def increase_by_one(s):
    for x in s:
        print(x)
        # if type(x)==dict:
        if type(s[x])==dict:
            print("d",x)
            increase_by_one(s[x])
            print(x)
        else:
            s[x]=s[x]+1
    return s
    f=[x for x in s]
    for x in f:
        s[x]=s[x]+1
    return increase_by_one(a)

def increase_by_one(s):
    f=[x for x in s]
    for x in f:
        s[x]=s[x]+1
    return s
print(increase_by_one({'1':2.7, '11':16, '111':{'a':5, 't':8}}))



def find_cpf_rate():
    i=0.0026314458525
    q=166000
    c=1280
    d=240
    p=deposit(q, i, 120)
    return i,balance(p, i, c, d)
def balance(p, i, c, d):
    return (p*(1+i)**d)-c*((1+i)**d-1)/i
def deposit(p, i, d):
    return p*(1+i)**d
print(find_cpf_rate())


def findx(x, a):
    try:
        # return str(a)+str([a.index(x)])
        return str([a.index(x)])
    except ValueError:
        for t,j in enumerate(a):
            if type(j)== list:
                # return str(a[t])+str(find_x(x, j))
                # return str(a)+str([t])+str([findx(x, j)])
                # return str([t])+str([findx(x, j)])
                return str([t])+findx(x, j)
        return

def findx(x, a):
    for i,j in enumerate(a):
        if type(j)== list:
            return str([i])+findx(x, j)
        if j is x:
            return str([i])
# print(find_x(5, [1, 5]))
# print(find_x(3, [1,5]))
print(find_x(5, [1, 3, [5], 3]))



class Number(object):
    # complete the class definition #
    def __init__(self, num):
        self.num=num
        # self.Undefined="Undefined"

    def plus(self,n):
        if 'Undefined' in [self.num,n.value()]:
            # return Undefined
            return Number('Undefined')
        else:
            return Number(self.num+n.value())

    def times(self,n):
        if 'Undefined' in [self.num,n.value()]:
            # return Undefined
            return Number('Undefined')
        else:
            return Number(self.num*n.value())

    def divide(self,n):
        if 'Undefined' in [self.num,n.value()]:
            # return Undefined
            return Number('Undefined')
        if n.value()==0:
            # return Undefined
            return Number('Undefined')
        else: return Number(self.num/n.value())

    def minus(self,n):
        if 'Undefined' in [self.num,n.value()]:
            # return Undefined
            return Number('Undefined')
        else:
            return Number(self.num-n.value())

    def value(self):
        if self.num=='Undefined': return "Undefined"
        return self.num

    def spell(self):
        if self.num=='Undefined': return "Undefined"
        elif len(str(self.num))>7: return "really large number"
        n=self.num

        a=str(n)
        b=list(reversed(a))

        one=['','one','two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten','eleven','twelve','thirteen' 'fourteen','fifteen']
        o=one[:-3]
        # twenty, thirty, fourty, fifty, sixty,seventy, eighty, ninety]
        if n<=15:
            return one[n]
        elif n<
        # print(one)
        # []
        ans=''
        for i,v in enumerate(b):
            # p=[' million', ' hundred','ty','thousand', 'hundred','teen']
            p=['','ty', ' hundred', ' thousand', 'ty', ' hundred', ' million']
            # plop=o[int(v)]+''+p[i]+' '
            plop=''+p[i]+' '
            if v=='0' and b[i+1]:
                plop=''
                # if :
            plopp=o[int(v)]+plop
            # print(i,i in [3])
            if i in [3]:
                plopp+='thousand '

            # if v=='1' and i in [4,1]:
            if i in [4,1]:
                if v=='1':
                    if int(v+b[i-1])<=12:
                        one[n]
                if =='0':
                    plopp='ten '
                elif:
                else:
                    plopp=

            if i in [4,1] and i<len(b):
                print(b[i+1],b[i+1]!='0')
                if b[i+1]!='0':
                    print('anded')
                    plopp='and '+plopp
            print(i,v)
            # ans+=o[int(v)]
            # ans=o[int(v)]+''+plop+''+ans
            ans=plopp+''+ans
        print(ans)
        # return [x for x in a]
gdn=Number(210792)
gdn.spell()










def pascal(row, col):
    if col == 1 or col == row:
        return 1
    else:
        return pascal(row - 1, col) + pascal(row - 1, col - 1)
col=2
a=[pascal(r, col) for r in range(1,5)]
print(a)




def bar(f, g):
    return lambda x: (lambda y: f(x))(g(x))
print( bar(lambda x:x+1, lambda x:x*2)(5)) #6?


s= [[0,1], [1,0], [1,1], [2,0]]
for x,y in s:
    print(x,y)
    y=3

for i,[x,y] in enumerate(s):
    print(x,y)
print(s)
for x in s:
    print(x)
print(s+'yf')

a = [1, [2, [3, 4]]]
b = a[1].copy()
c = b[1]
# c[0], b[0] = b[0], c[0]
b[0], c[0]=c[0], b[0]
print(a)
print(b)
print(c)

a=0
b=1
a,b=b,a
print(a,b)


a = [1, [1,2]]
b = a.copy()
# a[1] += [0] # both [1, [1, 2, 0]]
a[1] = [1,2,0] # a=[1, [1, 2, 0]] b=[1, [1, 2]]
a=[5]
b=a
a=[7]
a=a+[1]
print(id(a))
a+=[1] #affects b too
a.append(1) #affects b too
print(a)
print(b)
print(id(a))

a = ["CS", 1010, "U"]
b = [a[:2], "S"]
c = b.copy()
c[0][1] = 2020
# print (b + a[1])
print(a,b,c)


a=()
print(a)
# print(a[0],a[-1],a[0]+a[-1])
print(a[-1])

a=['a','b','A','B','Aa','AA','1','0','9','aa','ab',' ','a ','#']
print(sorted(a)) #[' ', '0', '1', '9', 'A', 'B', 'a', 'a ', 'aa', 'ab', 'b']

s = 'Lollapalooza'
d = {}
for i in range (len(s)): d[s[i%5]] = s[i]
print (d)
{'L': 'z', 'o': 'a', 'l': 'o', 'a': 'o'}

a=[1,2,3,4,5]
for x in a:
    # a+=[x] #inf loop
    a.pop()
print(a)

x = '1.2.3'
print(x[:]) #1.2.3
print(x[0:]) #1.2.3
print(x[:5]) #1.2.3
print(x[:-1]) #1.2.
print(x[1:0]) #

x, y, z = 0, 1, 2
def f(x):
    print(x)
    return g(x+y+z)
def g(y):
    print (y)
    return h(x+y+z)
def h(z) :
    print(z)
    return x+y+z
print (f(x+y+z)) #3 6 8 9

def f(x):
    print(x)
    def g(y):
        print (y)
        def h(z) :
            print(z)
            return x+y+z
        return h(x+y+z)
    return g(x+y+z)
print (f(x+y+z)) #3 6 11 20


