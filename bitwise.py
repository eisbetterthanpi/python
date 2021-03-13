# python "F:\bitwise.py"
# & | ^ ~   and or xor not
# p=False#True
# q=True
# print(~p&q)
#
# x='buffalo'
# exec("%s = %d" % (x,2))
# print(x)

sin="!pv!q"
# for x in sin if x not in "&|^~v":   #["&","|","^","~","and","or","not","!","v","(",")"]
# [x for x in sin if x not in "v"]

# signs="&|^~v"
signs=["&","|","^","~","and","or","not","!","v","(",")"]
vars=[x for x in sin if x not in signs]
# print(vars)
class Con:
    name=""
obj=Con()
# setattr(obj, vars[0], True)
# s=getattr(obj,vars[0])
# print(s)


for x in vars:
    setattr(obj, x, True)
    s=getattr(obj,vars[0])
# v! ~
# ^and &
# v or |
