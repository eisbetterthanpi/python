
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









