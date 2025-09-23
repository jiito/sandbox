# A basic implementation of MonteCarlo Tree Search


from abc import ABC, abstractmethod


class MCTSNode(ABC):
    pass

    @abstractmethod
    def expand():
        # TODO: add one or more child nodes
        pass

    def simulate():
        # TODO: run a low cost simulation of the game
        pass


def select():
    # TODO: implements the upper-confidence bound selection policy
    # UCB of node i = average reward of node i  + exploration term

    pass


class TicTacToeNode(MCTSNode):
    pass


def algo():
    """
    a four phase algo that first
    1. selects a path
    2. expands path
    3. simulates the expansion
    4. backpropogates stats to root node
    """

    pass


# TODO: implement tic-tac toe with this algorithm
