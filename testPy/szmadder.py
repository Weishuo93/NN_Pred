#! env python
class Adder(object):
    def __init__(self):
        print('calling constructor')
        self.info = 'default info'
    def add(self, a, b):
        print(f'Adder info is {self.info}')
        return a + b
