class ListNode:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class NodeGenerator:
    def __init__(self, *args):
        self.head = None
        for val in reversed(args):
            node = ListNode(val)
            node.next = self.head
            self.head = node

    def __iter__(self):
        current = self.head
        while current:
            yield current.val
            current = current.next


if __name__ == '__main__':
    generator = NodeGenerator(1, 2, 3, 4, 5)
    for val in generator:
        print(val)
