# A basic implementation of MonteCarlo Tree Search
from typing import List, Self


class Connect4:
    # Connect4 class copied from claude
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board: List[int] = [[0 for _ in range(cols)] for _ in range(rows)]
        self.current_player = 1  # Player 1 starts

    def make_move(self, col) -> bool:
        """Make a move in the specified column. Return True if valid."""
        # TODO: Implement move logic

        cols = self._board_columns()

        column_arr = cols[col]

        if self._col_is_full(column_arr):
            return False

        row = self._get_next_open_row(column_arr)
        self.board[row][col] = self.current_player

        self._change_player()

    def _board_columns(self):
        return [list(col) for col in zip(*self.board)]

    def _get_next_open_row(self, col: List[int]):
        try:
            return col.index(0)
        except ValueError:
            return -1

    def _col_is_full(self, col):
        return self._get_next_open_row(col) < 0

    def _change_player(self):
        self.current_player = 2 if self.current_player == 1 else 1

    def get_valid_moves(self):
        """Return list of valid column indices."""

        cols = self._board_columns()

        valid_cols = [i for i in range(len(cols)) if not self._col_is_full(cols[i])]

        wm = self.check_winning_move(valid_cols)

        if wm != -1:
            return [wm]

        return valid_cols

    def check_winning_move(self, moves: List[int]) -> int:
        # look for groups of 3 that can be completed SIKE
        # simulate each valid move and then if any are winning, return that move
        for move in moves:
            sim = self.copy()

            sim.make_move(move)

            w = sim.check_winner()
            if w == 1 or w == 2:
                return move

        return -1  # not a valid move

    def check_winner(self) -> int:
        """Return 1 if player 1 wins, 2 if player 2 wins, 0 if ongoing, -1 if draw."""

        result = 0
        result = self._check_horizontal_wins()
        if result > 0:
            return result
        result = self._check_vertical_wins()
        if result > 0:
            return result
        result = self._check_diagonal_wins()
        if result > 0:
            return result

        if sum([1 if self._col_is_full(c) else 0 for c in self._board_columns()]) == self.cols:
            result = -1

        return result

    def _check_horizontal_wins(self):
        for row in range(len(self.board)):
            consecutive_count = 1
            for j, c in enumerate(self.board[row][:-1]):
                if c == self.board[row][j + 1] and c != 0:
                    consecutive_count += 1
                else:
                    consecutive_count = 1

                if consecutive_count == 4:
                    return c
        return 0

    def _check_vertical_wins(self):
        cols = self._board_columns()
        for col in range(len(cols)):
            consecutive_count = 1
            for j, r in enumerate(cols[col][:-1]):
                if r == cols[col][j + 1] and r != 0:
                    consecutive_count += 1
                else:
                    consecutive_count = 1

                if consecutive_count == 4:
                    return r

        return 0

    def _check_diagonal_wins(self):
        for i, row in enumerate(self.board[:-3]):
            for j, col in enumerate(row[:-3]):
                if (
                    col
                    == self.board[i + 1][j + 1]
                    == self.board[i + 2][j + 2]
                    == self.board[i + 3][j + 3]
                    and col != 0
                ):
                    return col

        for i, row in enumerate(self.board[:-3]):
            for j, col in enumerate(row[::-1][:-3]):
                if (
                    col
                    == self.board[i + 1][::-1][j + 1]
                    == self.board[i + 2][::-1][j + 2]
                    == self.board[i + 3][::-1][j + 3]
                    and col != 0
                ):
                    return col

        return 0

    def reset(self):
        self.board: List[int] = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.current_player = 1  # Player 1 starts

    def copy(self) -> Self:
        """Return a deep copy of the current game state."""
        import copy

        return copy.deepcopy(self)

    def display(self):
        """Print the current board state."""

        for row in self.board[::-1]:
            print(f"| {'| '.join(['X' if s == 1 else ' ' if s == 0 else 'O' for s in row])}|")

        print("\n")


def test_connect_4():
    g = Connect4(6, 7)

    g.make_move(3)
    g.make_move(2)
    g.make_move(4)
    g.make_move(4)
    g.make_move(5)
    g.make_move(4)
    g.make_move(6)

    g.display()

    assert g.check_winner() == 1

    g.reset()

    g.make_move(1)
    g.make_move(3)
    g.make_move(2)
    g.make_move(4)
    g.make_move(4)
    g.make_move(5)
    g.make_move(4)
    g.make_move(6)

    g.display()

    assert g.check_winner() == 2

    g.reset()

    g.make_move(1)
    g.make_move(2)
    g.make_move(1)
    g.make_move(2)
    g.make_move(1)
    g.make_move(2)
    g.make_move(1)

    g.display()

    assert g.check_winner() == 1

    g.reset()

    g.make_move(1)
    g.make_move(2)
    g.make_move(2)
    g.make_move(3)
    g.make_move(4)
    g.make_move(3)
    g.make_move(3)
    g.make_move(5)
    g.make_move(4)
    g.make_move(4)
    g.make_move(4)

    g.display()

    assert g.check_winner() == 1

    g.reset()

    g.make_move(0)
    g.make_move(1)
    g.make_move(2)
    g.make_move(2)
    g.make_move(3)
    g.make_move(4)
    g.make_move(3)
    g.make_move(3)
    g.make_move(5)
    g.make_move(4)
    g.make_move(4)
    g.make_move(4)

    g.display()

    assert g.check_winner() == 2

    g.reset()

    g.make_move(5)
    g.make_move(4)
    g.make_move(4)
    g.make_move(3)
    g.make_move(2)
    g.make_move(3)
    g.make_move(3)
    g.make_move(5)
    g.make_move(2)
    g.make_move(2)
    g.make_move(2)

    g.display()

    assert g.check_winner() == 1

    g.reset()

    g.make_move(0)
    g.make_move(5)
    g.make_move(4)
    g.make_move(4)
    g.make_move(3)
    g.make_move(2)
    g.make_move(3)
    g.make_move(3)
    g.make_move(1)
    g.make_move(2)
    g.make_move(2)
    g.make_move(2)

    g.display()

    assert g.check_winner() == 2


if __name__ == "main":
    test_connect_4()
