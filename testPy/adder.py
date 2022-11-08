

class adder_class:
    def __init__(self):
        from time import time, ctime
        self.info = "test info:    " + str(ctime(time()))
        print("info value from py is:", self.info)

    def add(self, val1, val2):
        print("info value from py is:", self.info)
        return val1 + val2

    def printinfo(self):
        print("info value from py is:", self.info)
