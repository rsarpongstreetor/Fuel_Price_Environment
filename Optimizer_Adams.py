# @title  Optimizer
import torch.optim as optim

optimizer = None
if policy_network is not None and value_network is not None:
    # Get parameters from both networks
    all_params = list(policy_network.parameters()) + list(value_network.parameters())

    # Instantiate the Adam optimizer
    learning_rate = 3e-4 # Example learning rate
    optimizer = optim.Adam(all_params, lr=learning_rate)

    print(f"\nOptimizer instantiated with learning rate: {learning_rate}")
    print(optimizer)
else:
    print("\nPolicy or Value network not available. Cannot instantiate optimizer.")
