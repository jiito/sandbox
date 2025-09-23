from typing import List, Literal, Self


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


root = Node(0, [Node(1, [Node(3), Node(5)]), Node(2, [Node(2), Node(9)])])


assert tree_height(root) == 3
levels = level_order_traversal(root)
print(levels)


def minimax(root: Node, whose_turn: Literal["max", "min"]):
    # one agent tries to maximize the value , other agent tries to minimize
    # select depending on how to max
    if not root.children:
        return root.val

    # How would you derrive this?
    # what about trees that aren't perfect Binary Trees?

    # do we assume the max player goes first?

    # The algo is the maximum of the game on the subtrees, given the players turn...
    next_turn = "max" if whose_turn == "min" else "min"

    values = [minimax(c, next_turn) for c in root.children]

    return max(values) if whose_turn == "max" else min(values)


assert minimax(root, "max") == 3
assert minimax(root, "min") == 5
