import torch
import torch.nn as nn
import einops
import torch.nn.functional as F
import numpy as np
from jaxtyping import Float
from connect_4 import Connect4


class Connect4Net(nn.Module):
    def __init__(self, board_size=(6, 7), hidden_size=128):
        super(Connect4Net, self).__init__()
        self.board_size = board_size

        # TODO: Design your architecture
        # Input: board state representation
        # Output: policy probabilities + value estimate

        # Setup network
        # input shape: [row, cols, one_hot]

        rows, cols = board_size

        flattened_size = rows * cols

        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)

        self.policy_head = nn.Linear(hidden_size // 2, cols)

        self.value_head = nn.Linear(hidden_size // 2, 1)

    def forward(
        self, x: Float[torch.Tensor, "batch enc"]
    ) -> (Float[torch.Tensor, "batch cols"], Float[torch.Tensor, "batch value"]):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        policy_logits = self.policy_head(x)

        value_logit = self.value_head(x)

        return policy_logits, value_logit

    def predict(self, game_state: Connect4):
        """Convert game state to network input and get predictions"""
        # TODO: Convert Connect4 board to tensor
        # TODO: Run forward pass
        # TODO: Apply softmax to policy, mask invalid moves
        pass


def encode_board_state(game_state: Connect4) -> torch.Tensor:
    """Convert Connect4 board to neural network input"""
    # TODO: How will you represent the board?
    board = torch.tensor(game_state.board, dtype=torch.float)
    board = einops.rearrange(board, "r c -> (r c)")

    return board


def test_network_integration():
    game = Connect4()
    net = Connect4Net((game.rows, game.cols), 128)

    state_tensor = encode_board_state(game).unsqueeze(0)

    print(state_tensor.shape)
    assert state_tensor.shape == (1, game.rows * game.cols)

    policy_logits, value = net(state_tensor)

    valid_moves = game.get_valid_moves()

    mask = torch.full_like(policy_logits, float("-inf"))
    mask[0, valid_moves] = policy_logits[0, valid_moves]

    print(f"Policy shape: {policy_logits.shape}")
    print(f"Value: {value.shape}")
    print(f"mask: {mask}")


if __name__ == "__main__":
    test_network_integration()
