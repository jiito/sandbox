import math
import random
from typing import Dict, List, Tuple
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

            rand_move = random.choice(valid_moves)

            game.make_move(rand_move)
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


if __name__ == "__main__":
    # Test your simulator
    game = Connect4()
    game.make_move(3)
    game.make_move(2)
    game.make_move(4)
    game.make_move(0)
    game.make_move(5)

    game.display()

    simulator = MonteCarloSimulator(game)

    # Single rollout
    result = simulator.rollout(game)
    print(f"Random game result: {result}")

    # Position evaluation
    stats = simulator.evaluate_position(game, num_simulations=1000)
    print(f"Position evaluation: {stats}")

    # Best move suggestion
    best_move, evaluations = simulator.best_move_by_rollouts(game, num_simulations=500)
    print(f"Suggested move: column {best_move} for player {game.current_player}")
    print(f"Move evaluations: {evaluations}")
