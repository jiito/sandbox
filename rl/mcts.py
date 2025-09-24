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

            rand_col = math.floor(random.random() * len(valid_moves))

            game.make_move(rand_col)
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
        # TODO: Run multiple rollouts and aggregate results
        pass

    def best_move_by_rollouts(
        self, game_state: Connect4, num_simulations: int = 1000
    ) -> Tuple[int, Dict]:
        """
        Find the best move by comparing rollout results for each valid move.

        Returns:
            (best_column, move_evaluations)
        """
        # TODO: Try each valid move and compare average results
        pass


# Test your simulator
game = Connect4()
game.make_move(3)  # Player 1 plays center
game.make_move(2)  # Player 2 plays left-center

simulator = MonteCarloSimulator(game)

# Single rollout
result = simulator.rollout(game)
print(f"Random game result: {result}")
