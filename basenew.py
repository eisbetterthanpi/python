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
