# Written by Jason Koh

# Adopted from: http://stackoverflow.com/a/8412405
def rolling_window(l, w_size):
    for i in range(len(l)-w_size+1):
        yield [l[i+o] for o in range(w_size)]
