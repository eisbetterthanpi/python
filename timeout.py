# @title timeout

import time
def gen():
    time.sleep(1)
    return 5

import signal
import time

def timeout_handler(num, stack): raise Exception("timeout")
def long_function():
    time.sleep(2)
    return "done"

def rt():
    for i in range(3):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)
        try: return long_function()
        except Exception as ex: pass
        finally: signal.alarm(0)
print(rt())
