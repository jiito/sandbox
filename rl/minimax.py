from collections import deque
from typing import List, Self


class Node:
    children: List[Self]

    def __init__(self, val: int, children: List[Self | None] = []) -> None:
        self.val = val
        self.children = children

    def __repr__(self) -> str:
        return f"{self.val}"


def print_tree(root: Node):
    print(root.val)
    for c in root.children:
        print_tree(c)
    return


def level_order_traversal(root: Node):
    levels = [[root]]
    while True:
        curr_level = levels[-1]
        next_level = []
        for n in curr_level:
            print(n.val)
            next_level += n.children
        if len(next_level) > 0:
            levels.append(next_level)
        else:
            break
    return levels


def tree_height(root: Node) -> int:
    if not root.children:
        return 1

    return max([1 + tree_height(c) for c in root.children])


def minimax():
    pass


def test_minimax():
    pass


root = Node(0, [Node(1, [Node(3), Node(5)]), Node(2, [Node(2), Node(9)])])


print_tree(root)
print("=========")
print(f"Height of tree: {tree_height(root)}")
print("=========")
levels = level_order_traversal(root)
print(levels)
