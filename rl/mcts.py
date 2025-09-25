import math
import random
from typing import Dict, List, Optional, Tuple

from connect_4 import Connect4


class MonteCarloSimulator:
    def __init__(self, game: Connect4):
        self.game = game

    def rollout(self, game_state: Connect4, max_moves: int = 100) -> int:
        """
        Simulate a complete game from the current state using random moves.

        Args:
            game_state: Current Connect4 position to simulate from
            max_moves: Maximum moves to prevent infinite games

        Returns:
            Game result: 1 if player 1 wins, 2 if player 2 wins, -1 if draw
        """

        # copy game
        game = game_state.copy()

        for m in range(max_moves):
            valid_moves = game.get_valid_moves()
            if len(valid_moves) == 0:
                return -1

            move = random.choice(valid_moves)

            game.make_move(move)
            winner = game.check_winner()
            if winner > 0:
                return winner

        return -1

    def evaluate_position(
        self, game_state: Connect4, num_simulations: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate a position by running multiple rollouts.

        Returns:
            Dictionary with win rates: {'player1_wins': 0.4, 'player2_wins': 0.3, 'draws': 0.3}
        """

        win_counts = {1: 0, 2: 0, -1: 0}
        for i in range(num_simulations):
            winner = self.rollout(game)
            win_counts[winner] += 1

        results = {
            "player1_wins": win_counts[1] / num_simulations,
            "player2_wins": win_counts[2] / num_simulations,
            "draws": win_counts[-1] / num_simulations,
        }

        return results

    def best_move_by_rollouts(
        self, game_state: Connect4, num_simulations: int = 1000
    ) -> Tuple[int, Dict]:
        """
        Find the best move by comparing rollout results for each valid move.

        Returns:
            (best_column, move_evaluations)
        """
        # TODO: Try each valid move and compare average results
        valid_moves = game_state.get_valid_moves()
        current_player = game_state.current_player
        k = "player1_wins" if current_player == 1 else "player2_wins"

        best = (None, float("-inf"))
        evals_list = []
        for move in valid_moves:
            game = game_state.copy()

            game.make_move(move)

            results = self.evaluate_position(game, num_simulations)

            evals_list.append((move, results))

            if results[k] > best[1]:
                best = (move, results[k])

        move_evals = {move: results for move, results in evals_list}
        return best[0], move_evals


class MCTSNode:
    def __init__(
        self,
        game_state: Connect4,
        parent: Optional["MCTSNode"] = None,
        move_made: Optional[int] = None,
    ):
        self.game_state = game_state.copy()
        self.parent = parent
        self.move_made = move_made  # Move that led to this node

        # MCTS Statistics
        self.visits = 0
        self.wins = 0.0  # Can be fractional for draws

        # Tree Structure
        self.children: Dict[int, "MCTSNode"] = {}  # move -> child_node
        self.untried_moves = game_state.get_valid_moves()

        # player who made move is previous player
        self.player_who_made_move = 1 if game_state.current_player == 2 else 2

    def is_fully_expanded(self) -> bool:
        """Check if all possible moves have been tried."""
        # TODO: Return True if no untried moves remain

        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal game state."""
        # TODO: Return True if game is over (win/loss/draw)
        return self.game_state.check_winner() != 0

    def ucb1_value(self, exploration_param: float = 1.414) -> float:
        """Calculate UCB1 value for this node."""
        if self.visits == 0:
            return float("inf")  # Unvisited nodes get highest priority

        # UCB1 = win_rate + exploration_param * sqrt(ln(parent_visits) / visits)

        ucb1 = (self.wins / self.visits) + exploration_param * (
            math.log(self.parent.visits) / self.visits
        ) ** 0.5

        return ucb1

    def select_child(self, exploration_param: float = 1.414) -> "MCTSNode":
        """Select child with highest UCB1 value."""
        # TODO: Return child node with maximum UCB1 value
        max_child = (None, float("-inf"))
        for child in self.children.values():
            if child.ucb1_value(exploration_param) > max_child[1]:
                max_child = (child, child.ucb1_value(exploration_param))
        return max_child[0]

    def expand(self) -> "MCTSNode":
        """Create a new child node for an untried move."""
        # TODO:
        # 1. Pick a random untried move
        # 2. Create new game state by making that move
        # 3. Create child node
        # 4. Update untried_moves and children
        # 5. Return the new child

        move = random.choice(self.untried_moves)

        game = self.game_state.copy()

        game.make_move(move)

        self.children[move] = MCTSNode(game, self, move)

        self.untried_moves.remove(move)

        return self.children[move]

    def backpropagate(self, result: int):
        """Update statistics for this node and all ancestors."""
        # 1. Update visits and wins for this node
        # 2. Recursively update parent nodes
        # Note: Handle perspective - Player 1 win vs Player 2 win

        # key: 1 == player1 win, 2 == player2 win, -1 == draw

        # How do we keep track of who is winning?
        player = self.player_who_made_move

        if result == -1:
            self.wins += 0.5
        elif result == player:
            self.wins += 1

        self.visits += 1

        if self.parent:
            self.parent.backpropagate(result)


def plot_tree(root: MCTSNode):
    pass


if __name__ == "__main__":
    # Test your simulator
    game = Connect4()
    game.make_move(3)
    game.make_move(2)
    game.make_move(4)
    game.make_move(0)
    game.make_move(5)

    game.display()

    T = 5000  # timelimit

    t = 0

    root = MCTSNode(game)

    while t < T:
        current_node = root
        # select
        while current_node.is_fully_expanded() and not current_node.is_terminal():
            current_node = current_node.select_child()

        # expand
        if not current_node.is_terminal():
            current_node = current_node.expand()

        simulator = MonteCarloSimulator(current_node.game_state)

        # simulate
        result = simulator.rollout(current_node.game_state)

        # backprop
        current_node.backpropagate(result=result)

        t += 1
    print(f"moves: {[f'{k}, {v.ucb1_value()}' for k, v in root.children.items()]}")

    best_move = root.select_child(0)

    print(f"best move: {best_move.move_made}")

    # After your MCTS run, add this:
    print("\nDetailed node statistics:")
    for move, child in root.children.items():
        win_rate = child.wins / child.visits if child.visits > 0 else 0
        print(
            f"Move {move} by player {child.game_state.current_player} | {child.wins:.1f}/{child.visits} = {win_rate:.3f} win rate"
        )
        for move2, child2 in child.children.items():
            win_rate = child2.wins / child2.visits if child2.visits > 0 else 0
            print(
                f"\tMove {move2} by player {child2.game_state.current_player} | {child2.wins:.1f}/{child2.visits} = {win_rate:.3f} win rate"
            )
