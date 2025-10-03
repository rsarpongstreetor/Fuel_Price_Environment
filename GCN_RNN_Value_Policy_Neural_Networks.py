# @title Network Definitions

import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase # Import TensorDictModuleBase


class SimpleMultiAgentPolicy(TensorDictModuleBase): # Inherit from TensorDictModuleBase
     def __init__(self, input_x_dim, num_agents, num_individual_actions_features, num_action_categories, hidden_rnn_dim=64, in_keys=None, out_keys=None): # Added in_keys and out_keys
         super().__init__(in_keys=in_keys, out_keys=out_keys) # Pass keys to parent class
         self.num_agents = num_agents
         self.num_individual_actions_features = num_individual_actions_features
         self.num_action_categories = num_action_categories
         self.hidden_rnn_dim = hidden_rnn_dim

         # Define an RNN layer (e.g., GRU or LSTM)
         # Input to RNN: [seq_len, batch_size, input_size]
         # Here, seq_len is 1 (processing one time step at a time), batch_size is num_envs * num_agents, input_size is input_x_dim
         self.rnn = nn.GRU(input_x_dim, hidden_rnn_dim, batch_first=False) # batch_first=False for [seq_len, batch_size, input_size]

         # Linear layer after RNN
         # Input to linear layer: [batch_size, hidden_rnn_dim]
         # Output size: num_agents * num_individual_actions_features * num_action_categories
         self.linear = nn.Linear(hidden_rnn_dim * num_agents, # Input to linear layer is flattened hidden states from all agents
                                 num_agents * num_individual_actions_features * num_action_categories)

         # Initial hidden state for the RNN
         # This will be part of the state that needs to be carried over between steps by the collector
         self.register_buffer("rnn_hidden_state", torch.zeros(1, 1, self.hidden_rnn_dim)) # Shape: [num_layers * num_directions, batch_size, hidden_size]

     def forward(self, tensordict):
         # Input tensordict is expected to have ('agents', 'data', 'x') with shape [num_envs, num_agents, input_x_dim]
         # It might also contain the previous hidden state at a specific key (e.g., ('agents', 'rnn_hidden_state'))
         x = tensordict.get(('agents', 'data', 'x')) # Shape: [num_envs, num_agents, input_x_dim]
         prev_rnn_hidden_state = tensordict.get(('agents', 'rnn_hidden_state'), self.rnn_hidden_state.expand(1, x.shape[0] * self.num_agents, -1))

         if x is None:
              raise ValueError("Input tensordict does not contain ('agents', 'data', 'x') key.")

         # Reshape input for RNN: [seq_len, batch_size, input_size]
         # seq_len = 1
         # batch_size = num_envs * num_agents
         # input_size = input_x_dim
         rnn_input = x.view(1, -1, x.shape[-1]) # Shape: [1, num_envs * num_agents, input_x_dim]

         # Pass through RNN
         # rnn_output shape: [seq_len, batch_size, hidden_size] -> [1, num_envs * num_agents, hidden_rnn_dim]
         # next_rnn_hidden_state shape: [num_layers * num_directions, batch_size, hidden_size] -> [1, num_envs * num_agents, hidden_rnn_dim]
         rnn_output, next_rnn_hidden_state = self.rnn(rnn_input, prev_rnn_hidden_state)

         # Reshape RNN output to [num_envs, num_agents, hidden_rnn_dim]
         rnn_output_reshaped = rnn_output.view(tensordict.shape[0], self.num_agents, self.hidden_rnn_dim)

         # Flatten the reshaped RNN output for the linear layer
         # Flatten across the agent and hidden dimensions
         flattened_rnn_output = rnn_output_reshaped.view(tensordict.shape[0], -1) # Shape: [num_envs, num_agents * hidden_rnn_dim]


         # Get logits from the linear layer
         flattened_logits = self.linear(flattened_rnn_output) # Shape: [num_envs, num_agents * num_individual_actions_features * num_action_categories]


         # The ProbabilisticActor expects the module to return a tensordict with the out_keys,
         # and potentially any state that needs to be carried over (like RNN hidden state).
         # The value associated with the out_key ('agents', 'action') should be the logits.
         # The next RNN hidden state should be stored under a specific key,
         # e.g., ('agents', 'rnn_hidden_state'), so the collector can pass it in the next step.

         output_tensordict = TensorDict({
             ('agents', 'action'): flattened_logits, # Return flattened logits
             ('agents', 'rnn_hidden_state'): next_rnn_hidden_state # Store the next hidden state
         }, batch_size=tensordict.shape, device=x.device) # Use the input tensordict's batch size


         return output_tensordict


class SimpleMultiAgentValue(nn.Module):
    def __init__(self, input_x_dim, num_agents, hidden_rnn_dim=64):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_rnn_dim = hidden_rnn_dim

        # Define an RNN layer (e.g., GRU or LSTM)
        # Input to RNN: [seq_len, batch_size, input_size]
        # Here, seq_len is 1, batch_size is num_envs * num_agents, input_size is input_x_dim
        self.rnn = nn.GRU(input_x_dim, hidden_rnn_dim, batch_first=False)

        # Linear layer after RNN to output the value
        # If we want a single value per environment: Input size: num_agents * hidden_rnn_dim, Output size: 1
        # If we want a value per agent: Input size: hidden_rnn_dim, Output size: 1 (applied per agent)
        # Let's assume a single value per environment for simplicity initially.
        self.linear = nn.Linear(self.hidden_rnn_dim * self.num_agents, 1) # Output a single value per environment

        # Initial hidden state for the RNN
        self.register_buffer("rnn_hidden_state", torch.zeros(1, 1, self.hidden_rnn_dim))


    def forward(self, tensordict):
        # Input tensordict is expected to have ('agents', 'data', 'x') with shape [num_envs, num_agents, input_x_dim]
        # It might also contain the previous hidden state at a specific key (e.g., ('agents', 'rnn_hidden_state'))
        x = tensordict.get(('agents', 'data', 'x')) # Shape: [num_envs, num_agents, input_x_dim]
        prev_rnn_hidden_state = tensordict.get(('agents', 'rnn_hidden_state'), self.rnn_hidden_state.expand(1, x.shape[0] * self.num_agents, -1))

        if x is None:
             raise ValueError("Input tensordict does not contain ('agents', 'data', 'x') key.")

        # Reshape input for RNN: [seq_len, batch_size, input_size]
        rnn_input = x.view(1, -1, x.shape[-1]) # Shape: [1, num_envs * num_agents, input_x_dim]

        # Pass through RNN
        rnn_output, next_rnn_hidden_state = self.rnn(rnn_input, prev_rnn_hidden_state)

        # Reshape RNN output to [num_envs, num_agents, hidden_rnn_dim]
        rnn_output_reshaped = rnn_output.view(tensordict.shape[0], self.num_agents, self.hidden_rnn_dim)

        # Flatten the reshaped RNN output for the linear layer (for single value per env)
        flattened_rnn_output = rnn_output_reshaped.view(tensordict.shape[0], -1) # Shape: [num_envs, num_agents * hidden_rnn_dim]

        # Get the value from the linear layer
        value = self.linear(flattened_rnn_output) # Shape: [num_envs, 1]

        # The value network should return a tensordict with the value and the next hidden state.
        output_tensordict = TensorDict({
            'value': value, # Store the value
            ('agents', 'rnn_hidden_state'): next_rnn_hidden_state # Store the next hidden state
        }, batch_size=tensordict.shape, device=x.device)


        return output_tensordict

# Instantiate the value network
if env is not None:
    input_x_dim = env.node_feature_dim # Should be 1
    num_agents = env.num_agents # Should be 13
    hidden_rnn_dim = 64 # Match policy network hidden dim

    value_net = SimpleMultiAgentValue(
        input_x_dim=input_x_dim,
        num_agents=num_agents,
        hidden_rnn_dim=hidden_rnn_dim
    ).to(device)

    print(f"\nSimpleMultiAgentValue instantiated with input_x_dim={input_x_dim}, num_agents={num_agents}, hidden_rnn_dim={hidden_rnn_dim}.")
else:
    value_net = None
    print("\nEnvironment not instantiated, skipping value network instantiation.")
