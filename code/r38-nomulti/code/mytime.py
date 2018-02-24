import time

class Mytime(object):

    def __init__(self):
        self.zero()
        
    def zero(self):
        self.prev = time.time()

    def period(self):
        new = time.time()
        self.prev, gap = new, new - self.prev
        return gap
    
    def gap(self):
        return time.time() - self.prev
