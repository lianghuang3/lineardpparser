import time

class Mytime(object):

    __slots__ = "prev", "total"
    
    def __init__(self):
        self.zero()
        
    def zero(self):
        self.prev = time.time()
        self.total = 0

    def period(self):
        new = time.time()
        self.prev, gap = new, new - self.prev
        return gap
    
    def gap(self):
        return time.time() - self.prev

    def pause(self):
        new = time.time()
        self.total += new - self.prev
        
