# python "F:\gummy.py"
# import random
# start=100
# end=start
# bet=1
# mul=1
# for x in range(100):
#     win=random.randint(1,6)
#     if win in [1,6]: #win
#         end+=bet*mul
#         mul=1
#     else: #lose
#         end-=bet
#         mul+=0.5
#         if mul>3:
#             mul=3
# print('end',end)


def play(bet):
    start=100
    end=start
    bet=50
    mul=1
    for x in range(100):
        win=random.randint(1,6)
        if win in [1,2,6]: #win
            end+=bet*mul
            mul=1
        else: #lose
            end-=bet
            mul+=0.5
            if mul>3:
                mul=3
    return end


if __name__ == '__main__':
    import random
    import numpy as np
    result=[]
    for x in range(100):
        end=play(x)
        result.append(end)
    # print(result.mean())
    print(np.mean(result))
