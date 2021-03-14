# Created at 2020-06-23
# Summary: utils to profile memory
from memory_profiler import profile

@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

@profile
def func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    b = 0
    return a

if __name__ == '__main__':
    # my_func()
    func()