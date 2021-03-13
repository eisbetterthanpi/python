
# python "F:\itsalive.py"

# quine og
# a='a=%r;print(a%%a)';print(a%a)
# a='a=%r;exec(print(a%%a))';print(a%a)
# a='a=%r;exec(print(a%%a))';exec(a%a)
# print(a%a)
# work good
# exec(s:='print("exec(s:=%r)"%s)')
# exec(s:='print("exec(s:=%r)"%s)')


# exec(s:='print("exec(s:=%r)"%s)')

# nah
# exec(s:='exec(print("exec(s:=%r)"%s))')
# exec(s:='exec(print(s%s))')
# c=20

# # max recursion, no print
# ty='exec(%r)'
# s='exec(ty%s)'
# exec(s)



# exec(s:='print("exec(s:=%r)"%s)')
# great, changes but max recursion
# de='qwret'
# # de=de[2:]+de[:2]
# ty='print(%r);exec(%r);exec(%r)'
# # ty='print(exec(%r))'
# # s='print(ty%s)'
# s='exec(ty%(de,"de=de[2:]+de[:2]",s))'
# exec(s)


# exec(s:='exec("print(s)")')
# exec('s=\'exec("print(s)")\'\nprint(s)\nexec(s)')
# exec('s=\'exec(\\"print(s)\\")\nprint(s)\'\nexec(s)')

# exec(s:='exec("print(s)")\nprint(s)\nexec(s)')

# while True:print("its alive!")
# exec('while True:print("its alive!")')
# exec(s:='while True:print("its alive!")')


# exec('x = 10\nfor i in range(5):\n    if x%2 ==0: print(\'dgf\')\n    else: print(x)\n    x = x-1')

# s='exec(ty:='print(%r);exec(%r);exec(%r)'%(de,"de=de[2:]+de[:2]",s))'
# exec(ty:='print(%r);exec(%r);exec(%r)'%(de,"de=de[2:]+de[:2]",ty))

# ty='print(%r);exec(%r)'
# s='exec(ty%(s,s))'
# exec(s:='exec(ty%(s,s))'\n ty:='print(%r);exec(%r)')
# exec('s=\'exec(ty%(s,s))\'\n ty=\'print(%r);exec(%r)\'')





# python self replicating program cycle
# keyword de repeated
# exec(s:='print("exec(s:=%r)"%s)',de='qwret',de=de[2:]+de[:2])

# ert='print("exec(ert:=%r)"%ert)'
# qw=["zcvxb","adsvx","ndcv"]
# # exec(ert)
# de='qwret'
# de=de[2:]+de[:2]
# print("exec(ert:=%r)"%ert,)

# tu='df'
# print('hg%a'%tu) # hg'df'
# print('hg%s'%tu) # hgdf
# tu='tu=%r;print(tu%%tu)';print(tu%tu)

# tu='tu=%r;print(tu%%tu)'
# er='ret%re'
# print(tu%tu ,er%er)
# print(tu)
# print('ert'%tu)
# print(tu%'eryt') # tu='eryt';print(tu%tu)


# tu='tu=%s;print(tu%%tu);rty';print(tu%tu)



# s = 's = %r\ns = s.replace(s[-17],str(int(s[-17])+222))\nprint(s%%s)'
# s = s.replace(s[-17],str(int(s[-17])+222))
# print(s%s)
