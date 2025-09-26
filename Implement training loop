# @title Implement training loop
import torch
from tensordict import TensorDict
from torchrl.data import CompositeSpec, DiscreteTensorSpec
import torch.nn.utils as nn_utils # For gradient clipping
import random # For shuffling minibatches
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter for logging

# Initialize TensorBoard writer
log_dir = "/tmp/ppo_training_logs"
print(f"Setting up TensorBoard logging in directory: {log_dir}")
try:
    writer = SummaryWriter(log_dir)
    print("TensorBoard SummaryWriter initialized.")
except Exception as e:
    print(f"Error initializing TensorBoard SummaryWriter: {e}")
    writer = None


# Check if necessary objects are available
if collector is None:
    print("Error: Data collector is not available. Exiting training process.")
elif policy_network is None:
    print("Error: Policy network is not available. Exiting training process.")
elif value_network is None:
    print("Error: Value network is not available. Exiting training process.")
elif optimizer is None:
    print("Error: Optimizer is not available. Exiting training process.")
else:
    print("All necessary objects for training are available. Starting PPO training loop.")

    # 2. Define hyperparameters for the training loop
    num_training_iterations = 1000 # Total number of times to collect data and perform updates
    epochs_per_update = 10 # Number of times to iterate over collected data per training iteration
    minibatch_size = 64 # Size of minibatches for updates
    gamma = 0.99 # Discount factor for GAE
    lambda_ = 0.95 # GAE parameter
    clip_epsilon = 0.2 # Clipping parameter for policy loss
    value_loss_coef = 0.5 # Coefficient for value function loss
    entropy_coef = 0.01 # Coefficient for entropy bonus
    max_grad_norm = 1.0 # Maximum gradient norm for clipping
    checkpoint_interval = 100 # Save checkpoint every N iterations


    print("\nTraining Hyperparameters:")
    print(f"  num_training_iterations: {num_training_iterations}")
    print(f"  epochs_per_update: {epochs_per_update}")
    print(f"  minibatch_size: {minibatch_size}")
    print(f"  gamma: {gamma}")
    print(f"  lambda_: {lambda_}")
    print(f"  clip_epsilon: {clip_epsilon}")
    print(f"  value_loss_coef: {value_loss_coef}")
    print(f"  entropy_coef: {entropy_coef}")
    print(f"  max_grad_norm: {max_grad_norm}")
    print(f"  checkpoint_interval: {checkpoint_interval}")


    # 3. Start the main training loop
    for i in range(num_training_iterations):
        print(f"\n--- Training Iteration {i+1}/{num_training_iterations} ---")

        # 4. Collect data using the collector
        print("Collecting data...")
        try:
            # The collector will yield a TensorDict containing 'frames_per_batch' worth of data
            # which corresponds to one step across all environments (num_envs).
            # The collector is typically configured to collect a full rollout of T steps.
            # Let's assume the collector is set up to collect `rollout_len` steps per environment
            # resulting in a TensorDict with batch shape [num_envs, rollout_len].
            # The total frames collected per iteration will be num_envs * rollout_len.

            # Assuming the collector is configured to yield a single large tensordict per iteration
            # containing the full rollout data.
            # The data will have batch shape [num_envs, rollout_len] after collection.
            collected_data = collector.next()
            print(f"Data collected. Shape: {collected_data.shape}")
            # print(f"Data keys: {collected_data.keys(include_nested=True)}") # Debug print

        except Exception as e:
            print(f"An error occurred during data collection: {e}")
            # Log error if writer is available
            if writer is not None:
                writer.add_text('Errors', f'Data collection failed at iteration {i+1}: {e}', i)
            continue # Skip to the next iteration if data collection fails

        # 5. Ensure the collected TensorDict is on the correct device
        collected_data = collected_data.to(device)
        # print(f"Data moved to device: {collected_data.device}") # Debug print

        # 6. Extract relevant data from the collected TensorDict
        # Expected structure: collected_data has batch shape [B, T] (where B=num_envs, T=rollout_len)
        # Inside, keys like 'reward', 'done', 'action' are at the root or nested.
        # 'value' and 'log_prob' from the OLD policy are expected under ('old_policy', ...)
        # 'next.value' from the OLD policy is expected under ('next', ('old_policy', ...))

        try:
            rewards = collected_data['reward'] # Shape [B, T, num_agents, 1] (assuming agent dim)
            old_values = collected_data[('old_policy', 'value')] # Shape [B, T, num_agents, 1] (assuming agent dim)
            done = collected_data['done'] # Shape [B, T, 1] (episode termination)
            old_log_probs = collected_data[('old_policy', 'log_prob')] # Shape [B, T, 1] (log prob of joint action)
            actions = collected_data['action'] # Shape [B, T, num_agents, num_individual_actions_features] (action taken)
            next_old_values = collected_data['next'][('old_policy', 'value')] # Shape [B, T, num_agents, 1] (V(s_{t+1}) for step t)

            # print(f"Extracted data shapes:") # Debug print
            # print(f"  rewards: {rewards.shape}")
            # print(f"  old_values: {old_values.shape}")
            # print(f"  done: {done.shape}")
            # print(f"  old_log_probs: {old_log_probs.shape}")
            # print(f"  actions: {actions.shape}")
            # print(f"  next_old_values: {next_old_values.shape}")


        except KeyError as e:
            print(f"Error extracting data from tensordict: Missing key {e}. Please ensure the collector populates the tensordict with the required keys.")
            if writer is not None:
                writer.add_text('Errors', f'Data extraction failed at iteration {i+1}: Missing key {e}', i)
            continue
        except Exception as e:
            print(f"An unexpected error occurred during data extraction: {e}")
            if writer is not None:
                writer.add_text('Errors', f'Unexpected data extraction error at iteration {i+1}: {e}', i)
            continue


        # 7. Compute the advantage estimates using GAE
        # The compute_gae function expects rewards, values, done with a time dimension at the end.
        # Our tensors have shape [B, T, num_agents, 1].
        # GAE is computed independently for each agent in each environment.
        # Reshape from [B, T, num_agents, 1] to [B * num_agents, T, 1] for GAE computation.

        # Need to handle the 'done' tensor shape. If it's [B, T, 1], it needs to be broadcast or repeated
        # to match the agent dimension for the flattened GAE calculation.
        # Assuming 'done' is environment-wise, not agent-wise.
        # Repeat 'done' along the agent dimension: [B, T, 1] -> [B, T, num_agents, 1]
        if done.shape[-2] == 1 and old_values.shape[-2] > 1:
             done_for_gae = done.repeat(1, 1, old_values.shape[-2], 1)
        elif done.shape[-2] == old_values.shape[-2]:
             done_for_gae = done
        else:
             print(f"Error: done shape {done.shape} is not compatible with value shape {old_values.shape} for GAE.")
             if writer is not None:
                writer.add_text('Errors', f'Shape mismatch for done and values in GAE at iteration {i+1}.', i)
             continue

        # Flatten dimensions for GAE calculation: [B, T, num_agents, 1] -> [B * num_agents, T, 1]
        B, T, num_agents, _ = old_values.shape
        flat_rewards = rewards.view(B * num_agents, T, 1)
        flat_old_values = old_values.view(B * num_agents, T, 1)
        flat_done_for_gae = done_for_gae.view(B * num_agents, T, 1)


        print("Computing advantages...")
        try:
            # Pass flat tensors to compute_gae
            advantages = compute_gae(flat_rewards, flat_old_values, gamma, lambda_, flat_done_for_gae) # Shape [B * num_agents, T, 1]

            # Reshape advantages back to original structure [B, T, num_agents, 1]
            advantages = advantages.view(B, T, num_agents, 1)
            # print(f"Advantages shape: {advantages.shape}") # Debug print

        except Exception as e:
            print(f"An error occurred during advantage computation: {e}")
            if writer is not None:
                writer.add_text('Errors', f'Advantage computation failed at iteration {i+1}: {e}', i)
            continue

        # 8. Compute target values for the value function loss (GAE + old_values)
        target_values = advantages + old_values # Shape [B, T, num_agents, 1]
        # print(f"Target values shape: {target_values.shape}") # Debug print


        # 9. Flatten the collected data along the batch and time dimensions for minibatch processing
        # Reshape from [B, T, ...] to [B*T, ...] for relevant tensors
        # Need: observations ('agents', 'data', 'x'), actions, old_log_probs, advantages, target_values
        # Observations shape: [B, T, num_agents, node_feature_dim]
        # Actions shape: [B, T, num_agents, num_individual_actions_features]
        # old_log_probs shape: [B, T, 1]
        # advantages shape: [B, T, num_agents, 1]
        # target_values shape: [B, T, num_agents, 1]

        # Flatten batch and time dimensions together
        # For observations and actions, the agent dimension remains as part of the 'event'
        # For log_probs, advantages, target_values, the last dimension is 1 or num_agents
        # Need to be careful with dimensions when flattening and creating minibatches.

        # Let's flatten to [B * T, ...]
        # Need node_feature_dim to reshape flat_obs_x
        try:
            node_feature_dim = env.node_feature_dim # Get node_feature_dim from env
            num_individual_actions_features = env.num_individual_actions_features # Get from env

            flat_obs_x = collected_data[('agents', 'data', 'x')].view(B * T, num_agents, node_feature_dim)
            flat_actions = actions.view(B * T, num_agents, num_individual_actions_features)
            flat_old_log_probs = old_log_probs.view(B * T, 1) # Log probs are already summed over agents
            # Need to handle advantages and target values - they are per agent.
            # Flattening them to [B*T, num_agents, 1] is appropriate for per-agent value loss.
            flat_advantages = advantages.view(B * T, num_agents, 1)
            flat_target_values = target_values.view(B * T, num_agents, 1)

            # Create a flattened TensorDict for minibatching
            # This TensorDict will have batch size B*T
            flat_tensordict = TensorDict({
                ('agents', 'data', 'x'): flat_obs_x,
                'action': flat_actions,
                ('old_policy', 'log_prob'): flat_old_log_probs,
                'advantages': flat_advantages, # Store advantages for policy loss
                'target_values': flat_target_values, # Store target values for value loss
                # Include other necessary keys if policy/value modules require them
            }, batch_size=[B * T]).to(device) # Ensure the new tensordict is on the correct device

        except Exception as e:
             print(f"An error occurred during data flattening for minibatching: {e}")
             if writer is not None:
                writer.add_text('Errors', f'Data flattening failed at iteration {i+1}: {e}', i)
             continue


        # 10. Iterate over multiple epochs for policy and value network updates (as in previous block)
        print(f"Starting {epochs_per_update} epochs of updates...")
        for epoch in range(epochs_per_update):
            # 11. Within each epoch, shuffle the flattened data and iterate over minibatches (as in previous block)
            indices = list(range(B * T))
            random.shuffle(indices)

            print(f"  Epoch {epoch+1}/{epochs_per_update}")

            for start_idx in range(0, B * T, minibatch_size):
                end_idx = min(start_idx + minibatch_size, B * T)
                minibatch_indices = indices[start_idx:end_idx]

                # 12. For each minibatch, create a minibatch TensorDict (as in previous block)
                minibatch_tensordict = flat_tensordict[minibatch_indices]

                # print(f"    Processing minibatch with {len(minibatch_indices)} samples.") # Debug print
                # print(f"    Minibatch keys: {minibatch_tensordict.keys(include_nested=True)}") # Debug print


                # 13. Zero the gradients of the optimizer (as in previous block)
                optimizer.zero_grad()

                # 14. Compute the total PPO loss for the minibatch
                try:
                    # Forward pass through policy and value networks using the minibatch
                    minibatch_tensordict_processed = minibatch_tensordict.clone() # Clone to avoid modifying original minibatch
                    policy_network(minibatch_tensordict_processed) # Adds new_log_probs and action_dist
                    value_network(minibatch_tensordict_processed) # Adds new_values

                    # Compute individual loss components for logging
                    # Policy Loss
                    # Get new_log_probs from the processed tensordict
                    # Ensure new_log_probs match shape [minibatch_size, 1] for policy loss calculation
                    # MultiCategorical.log_prob(action) returns [..., 1]
                    # Action shape in minibatch_tensordict is [minibatch_size, num_agents, num_individual_actions_features]
                    # Policy distribution from processed tensordict is 'action_dist'
                    new_log_probs_minibatch = minibatch_tensordict_processed['action_dist'].log_prob(minibatch_tensordict['action']).unsqueeze(-1)
                    # old_log_probs for minibatch are in minibatch_tensordict[('old_policy', 'log_prob')], shape [minibatch_size, 1]
                    # advantages for minibatch are in minibatch_tensordict['advantages'], shape [minibatch_size, num_agents, 1]
                    # Need mean advantages for policy loss, shape [minibatch_size, 1]
                    mean_advantages_minibatch = torch.mean(minibatch_tensordict['advantages'], dim=-2, keepdim=True)

                    policy_loss = compute_policy_loss(
                        new_log_probs_minibatch,
                        minibatch_tensordict[('old_policy', 'log_prob')],
                        mean_advantages_minibatch,
                        clip_epsilon
                    )

                    # Value Loss
                    # Get new_values from the processed tensordict, shape [minibatch_size, num_agents, 1]
                    new_values_minibatch = minibatch_tensordict_processed[('agents', 'state_value')]
                    # target_values for minibatch are in minibatch_tensordict['target_values'], shape [minibatch_size, num_agents, 1]
                    value_loss = compute_value_loss(
                        new_values_minibatch,
                        minibatch_tensordict['target_values']
                    )

                    # Entropy Bonus
                    policy_distribution_minibatch = minibatch_tensordict_processed['action_dist']
                    entropy_bonus = compute_entropy_bonus(policy_distribution_minibatch)

                    # Total Loss (combined)
                    total_loss = compute_total_ppo_loss(policy_loss, value_loss, entropy_bonus, value_loss_coef, entropy_coef)

                    # Log metrics per minibatch
                    if writer is not None:
                         global_step = i * epochs_per_update * (B * T // minibatch_size) + epoch * (B * T // minibatch_size) + (start_idx // minibatch_size)
                         writer.add_scalar('Loss/Total', total_loss.item(), global_step)
                         writer.add_scalar('Loss/Policy', policy_loss.item(), global_step)
                         writer.add_scalar('Loss/Value', value_loss.item(), global_step)
                         writer.add_scalar('Loss/Entropy', entropy_bonus.item(), global_step)
                         # Add other metrics if available (e.g., explained variance)


                except Exception as e:
                    print(f"An error occurred during PPO loss computation for minibatch: {e}")
                    if writer is not None:
                         global_step = i * epochs_per_update * (B * T // minibatch_size) + epoch * (B * T // minibatch_size) + (start_idx // minibatch_size)
                         writer.add_text('Errors', f'Minibatch loss computation failed at step {global_step}: {e}', global_step)
                    continue # Skip to the next minibatch

                # 15. Perform a backward pass to compute gradients (as in previous block)
                total_loss.backward()

                # 16. Clip the gradients (as in previous block)
                if max_grad_norm is not None:
                    # Clip gradients for policy and value network parameters
                    all_minibatch_params = list(policy_network.parameters()) + list(value_network.parameters())
                    nn_utils.clip_grad_norm_(all_minibatch_params, max_grad_norm)


                # 17. Step the optimizer to update the network parameters (as in previous block)
                optimizer.step()

            # 18. After iterating through all minibatches in an epoch, continue to the next epoch.
            # Epoch finished.

        # 19. After completing all epochs for the current training iteration, reset the collector.
        # The collector automatically resets after yielding its total_frames (or frames_per_batch if total_frames is None).
        # If the collector is configured to collect a fixed number of steps per iteration,
        # calling collector.next() again will start a new collection phase.
        # No explicit reset needed here if using the collector in a loop like this.

        # 20. Log metrics per training iteration
        if writer is not None:
             iteration_global_step = i
             # Calculate and log episode rewards
             # Need to iterate through the collected_data and sum rewards for episodes ending in this batch.
             # Handle multiple environments and episodes within the batch.
             # 'done' tensor shape is [B, T, 1].
             # 'reward' tensor shape is [B, T, num_agents, 1].

             episode_rewards = []
             # Iterate through each environment in the batch
             for env_idx in range(B):
                  current_episode_reward = 0.0
                  # Iterate through time steps for this environment
                  for t in range(T):
                       # Sum rewards for all agents at this step
                       step_reward = rewards[env_idx, t].sum().item() # Sum over agents, get scalar
                       current_episode_reward += step_reward

                       # Check if the episode ended at this step
                       if done[env_idx, t].item() == 1:
                            # Episode ended, log the total reward for this episode
                            episode_rewards.append(current_episode_reward)
                            # Reset for the next episode (if any starts in this batch)
                            current_episode_reward = 0.0

                  # If the rollout ended but the episode didn't, the partial episode reward is not logged here.
                  # This is a common simplification; full episode rewards are typically logged only when 'done' is true.
                  # Or, if the collector tracks episode stats, use those.

             if episode_rewards:
                  mean_episode_reward = sum(episode_rewards) / len(episode_rewards)
                  writer.add_scalar('Reward/MeanEpisodeReward', mean_episode_reward, iteration_global_step)
                  print(f"  Logged mean episode reward: {mean_episode_reward:.2f}")
             else:
                  print("  No episodes finished in this iteration's rollout.")


        # 21. Periodically save model checkpoints
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_path = f"/tmp/ppo_checkpoint_iter_{i+1}.pt"
            print(f"Saving checkpoint to {checkpoint_path}...")
            try:
                torch.save({
                    'iteration': i + 1,
                    'policy_state_dict': policy_network.state_dict(),
                    'value_state_dict': value_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # Save other relevant information like hyperparameters
                }, checkpoint_path)
                print("Checkpoint saved.")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")


    print("\nTraining loop finished.")
    # 22. Close the TensorBoard writer after the loop
    if writer is not None:
        print("Closing TensorBoard SummaryWriter.")
        writer.close()
