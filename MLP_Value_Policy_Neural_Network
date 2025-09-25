import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import MLP # Using MLP for convenience, or could use nn.Sequential

# Assuming FlattenObservationTransformer is defined in a previous cell

class PolicyNetwork(nn.Module):
    """
    Neural network for the policy function in PPO.
    Takes flattened observations and outputs logits for the MultiCategorical distribution.
    """
    def __init__(self, num_agents, node_feature_dim, num_individual_actions_features, num_action_categories):
        super().__init__()
        self.num_agents = num_agents
        self.node_feature_dim = node_feature_dim
        self.num_individual_actions_features = num_individual_actions_features
        self.num_action_categories = num_action_categories

        # Input size: flattened observation [num_agents * node_feature_dim]
        input_size = num_agents * node_feature_dim
        # Output size: flattened logits [num_agents * num_individual_actions_features * num_action_categories]
        output_size = num_agents * num_individual_actions_features * num_action_categories

        # Use FlattenObservationTransformer as the first step
        self.flatten_obs = FlattenObservationTransformer(
            num_agents=num_agents,
            node_feature_dim=node_feature_dim,
            in_keys=[('agents', 'data', 'x')],
            out_keys=[('agents', 'data', 'x_flattened')]
        )

        # Define the main network layers
        # Example using MLP for hidden layers, followed by a linear layer for logits
        # The MLP takes flattened input and outputs to a hidden layer size,
        # then a final linear layer maps to the flattened logits size.
        hidden_size = 128 # Example hidden layer size
        self.mlp = MLP(
            in_features=input_size,
            out_features=hidden_size,
            num_cells=[hidden_size], # One hidden layer
            activation_class=nn.ReLU,
        )

        self.logits_layer = nn.Linear(hidden_size, output_size)


    def forward(self, tensordict):
        # Apply the observation flattening transformer
        # This adds ('agents', 'data', 'x_flattened') to the tensordict
        tensordict = self.flatten_obs(tensordict)

        # Get the flattened observation
        x_flat = tensordict.get(('agents', 'data', 'x_flattened'))

        # Pass through the network layers
        # The input to MLP should be [..., flattened_input_size]
        # Flattened_obs shape: [..., num_agents * node_feature_dim]
        # MLP expects the last dimension to be features.
        # Batch shape is ... (potentially [B, T])
        # Reshape x_flat from [..., num_agents * node_feature_dim] to [Product(batch_shape), num_agents * node_feature_dim]
        original_shape = x_flat.shape[:-1]
        x_flat_reshaped = x_flat.view(-1, x_flat.shape[-1]) # Flatten batch dims

        mlp_output = self.mlp(x_flat_reshaped) # Shape [Product(batch_shape), hidden_size]
        logits_flat = self.logits_layer(mlp_output) # Shape [Product(batch_shape), output_size]

        # Reshape logits back to original batch shape + output size
        logits = logits_flat.view(original_shape + torch.Size([self.num_agents * self.num_individual_actions_features * self.num_action_categories]))

        # Store logits in the tensordict under the specified key
        tensordict[('policy', 'logits')] = logits

        return tensordict


class ValueNetwork(nn.Module):
    """
    Neural network for the value function in PPO.
    Takes flattened observations and outputs a value estimate per agent.
    """
    def __init__(self, num_agents, node_feature_dim):
        super().__init__()
        self.num_agents = num_agents
        self.node_feature_dim = node_feature_dim

        # Input size: flattened observation [num_agents * node_feature_dim]
        input_size = num_agents * node_feature_dim
        # Output size: value estimate per agent [num_agents]
        # The PPO loss typically expects value predictions of shape [..., num_agents, 1]
        # So the final linear layer should output num_agents * 1 features.
        output_size = num_agents * 1

        # Use FlattenObservationTransformer as the first step
        self.flatten_obs = FlattenObservationTransformer(
            num_agents=num_agents,
            node_feature_dim=node_feature_dim,
            in_keys=[('agents', 'data', 'x')],
            out_keys=[('agents', 'data', 'x_flattened')]
        )

        # Define the main network layers
        # Example using MLP for hidden layers, followed by a linear layer for value estimates
        hidden_size = 128 # Example hidden layer size
        self.mlp = MLP(
            in_features=input_size,
            out_features=hidden_size,
            num_cells=[hidden_size], # One hidden layer
            activation_class=nn.ReLU,
        )

        self.value_layer = nn.Linear(hidden_size, output_size)

    def forward(self, tensordict):
        # Apply the observation flattening transformer
        # This adds ('agents', 'data', 'x_flattened') to the tensordict
        tensordict = self.flatten_obs(tensordict)

        # Get the flattened observation
        x_flat = tensordict.get(('agents', 'data', 'x_flattened'))

        # Pass through the network layers
        # Reshape x_flat from [..., num_agents * node_feature_dim] to [Product(batch_shape), num_agents * node_feature_dim]
        original_shape = x_flat.shape[:-1]
        x_flat_reshaped = x_flat.view(-1, x_flat.shape[-1]) # Flatten batch dims

        mlp_output = self.mlp(x_flat_reshaped) # Shape [Product(batch_shape), hidden_size]
        values_flat = self.value_layer(mlp_output) # Shape [Product(batch_shape), num_agents * 1]

        # Reshape values back to original batch shape + [num_agents, 1]
        values = values_flat.view(original_shape + torch.Size([self.num_agents, 1]))

        # Store value estimates in the tensordict under the specified key
        tensordict[('agents', 'state_value')] = values

        return tensordict
