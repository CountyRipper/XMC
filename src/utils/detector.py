from functools import wraps

def log(func):
    @wraps(func)
    def  wrapper(*args,**kw):
        print("call %s():" % func.__name__)
        
        for i in args:
            print(i)
        return func(*args, **kw)
    return wrapper