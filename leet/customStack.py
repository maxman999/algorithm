class ListNode:
    def __init__(self, val=None, next=None, prev=None):
        self.val = val
        self.prev = prev
        self.next = next


class Stack:
    head = None

    def push(self, val):
        if not self.head:
            self.head = ListNode(val)
        else:
            temp = self.head
            self.head = ListNode(val)
            self.head.next = temp

    def pop(self):
        if self.head:
            target = self.head
            self.head = self.head.next
            return target.val
        return None


class MyCircularDeque:
    def __init__(self, size):
        self.size = size
        self.head = None
        self.tail = None

    def insertFront(self, val):
        if not self.head:
            self.head = ListNode(val)
            self.tail = self.head
        else:
            tmp = self.head
            self.head = ListNode(val)
            self.head.next, self.tail = tmp
            self.head.prev = self.tail

    def insertLast(self, val):
        return

    # def deleteFront(self):

    # def deleteLast(self):


if __name__ == '__main__':
    circular_deque = MyCircularDeque(5)

    circular_deque.insertFront(2)
    circular_deque.insertFront(3)
    circular_deque.insertFront(4)
    circular_deque.insertFront(5)
    # circular_deque.insertLast(5)


