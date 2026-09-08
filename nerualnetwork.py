import torch
import torch.nn as nn


class PointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

    def forward(self, points):
        # Encode each point, then keep the strongest feature across the workspace.
        point_features = self.layers(points)
        return point_features.max(dim=1).values


class RobotMotionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = PointEncoder()
        self.policy = nn.Sequential(
            nn.Linear(256 + 3 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh(),
        )

    def forward(self, workspace_points, robot_state, object_position):
        workspace_features = self.point_encoder(workspace_points)
        combined_state = torch.cat(
            (workspace_features, robot_state, object_position), dim=1
        )
        return self.policy(combined_state)


robot_motion_network = RobotMotionNetwork()

# Example shape check. Replace these tensors with your real normalized data.
example_points = torch.randn(1, 1000, 2)
example_robot_state = torch.randn(1, 3)       # x, y, rotation
example_object_position = torch.randn(1, 2)   # x, y
example_action = robot_motion_network(
    example_points,
    example_robot_state,
    example_object_position,
)

print("Predicted action [delta_x, delta_y, delta_rotation]:", example_action)
print("Output shape:", tuple(example_action.shape))