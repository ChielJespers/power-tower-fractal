MAX_ITER = 500

def powertower(c):
    z = c
    n = 0
    while n < MAX_ITER:
        try:
            z = c**z
        except OverflowError as E:
            return n
        except ZeroDivisionError as D:
            return MAX_ITER
        n += 1
    return n