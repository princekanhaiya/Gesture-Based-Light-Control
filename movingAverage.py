from collections import deque

#thinking process
#because the element is coming one after another, so when the size + 1's number coming in, we need to remove the first coming number from the window and calculate the avg
#so a queue can be used here to ensure the first in first out
#keep a queue size <= k, if greater than k then pop and substract the value that poped out and add the number that come in and calculate the avg



class MovingAverage:
    """
    @param: size: An integer
    """
    def __init__(self, size):
        # do intialization if necessary
        self.que = deque([])
        self.size = size
        self.total = 0.0

    """
    @param: val: An integer
    @return:  
    """
    def next(self, val):
        # write your code here
        if len(self.que) < self.size:
            self.total += val
            self.que.appendleft(val)
        else:
            self.total -= self.que.pop()
            self.que.appendleft(val)
            self.total += val

        return self.total / len(self.que)



# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param = obj.next(val)