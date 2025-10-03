# @title 2. Evaluate Trained Policy 
# Assuming 'env', 'policy_network', and 'device' are available from previous steps
# Assuming 'AnFuelpriceEnv' class is defined and available

eval_env = None
eval_policy = None

if env is not None and policy_network is not None:
    print("\nSetting up evaluation environment...")
    try:
        # Create a single environment for evaluation (num_envs=1)
        # Use the same seed or a different one for evaluation
        eval_seed = 42 # Use a fixed seed for reproducibility
        eval_env = AnFuelpriceEnv(num_envs=1, seed=eval_seed, device=device, episode_length=env.episode_length, allow_repeat_data=False) # Typically don't allow repeat data in eval

        print("Evaluation environment instantiated successfully.")

        # For evaluation, we typically use the deterministic mode of the policy
        # We can wrap the policy_network in a ProbabilisticActor and set it to deterministic mode,
        # or if the policy network itself has a deterministic mode, use that.
        # Assuming the policy_network is the module within a ProbabilisticActor,
        # we can create a new ProbabilisticActor for evaluation or set the existing one's mode.

        # If you have the ProbabilisticActor instance 'policy' from previous steps:
        if 'policy' in locals() and policy is not None:
             eval_policy = policy.clone() # Clone the policy
             eval_policy.set_deterministic_mode(True) # Set to deterministic mode

        # If you only have the policy_network module:
        else:
            print("ProbabilisticActor 'policy' instance not found. Creating a new one for evaluation.")
            # Need action spec to create ProbabilisticActor
            # Assuming action spec is available from the evaluation environment
            eval_action_spec_composite = eval_env.action_spec[('agents', 'action')]
            eval_action_spec = eval_action_spec_composite[('agents', 'action')]

            # Need the maker function
            # Assuming multi_categorical_maker() is defined and available
            if 'multi_categorical_maker' in locals():
                 eval_policy_module = policy_network # Use the trained policy network module
                 eval_policy = ProbabilisticActor(
                     module=eval_policy_module,
                     in_keys=[('agents', 'data')], # Match input keys used during training
                     out_keys=[('agents', 'action')], # Match output keys used during training
                     distribution_class=multi_categorical_maker(), # Use the maker function
                     return_log_prob=False, # No need for log_prob during evaluation
                 ).to(device)
                 eval_policy.set_deterministic_mode(True) # Set to deterministic mode
            else:
                 print("Error: multi_categorical_maker function not found. Cannot create evaluation policy.")
                 eval_policy = None


        print("Evaluation policy set up (in deterministic mode).")

    except Exception as e:
        print(f"\nAn error occurred during evaluation setup: {e}")
        eval_env = None
        eval_policy = None

else:
    print("\nEnvironment or policy network not available. Cannot set up evaluation.")

# Run a simple evaluation loop
if eval_env is not None and eval_policy is not None:
    print("\nRunning evaluation episodes...")
    num_eval_episodes = 5 # Number of episodes to run for evaluation
    episode_rewards = []

    try:
        for episode in range(num_eval_episodes):
            print(f"  Running evaluation episode {episode + 1}/{num_eval_episodes}...")
            # Reset the evaluation environment
            eval_tensordict = eval_env.reset()
            done = eval_tensordict['done'] # Get initial done flag

            total_episode_reward = 0.0

            # Run the episode until it's done or truncated
            while not done.all():
                # Get action from the evaluation policy (in deterministic mode)
                # The policy expects a tensordict with the current state
                # The policy will add the action to the tensordict
                with torch.no_grad(): # No gradient calculation during evaluation
                    eval_tensordict = eval_policy(eval_tensordict)

                # Step the environment with the sampled action
                # The environment returns the next state and reward
                eval_tensordict = eval_env.step(eval_tensordict)

                # Accumulate reward (assuming agent-wise reward at ('agents', 'reward'))
                # The reward in the stepped tensordict is for the transition *to* the current state
                # and is available under 'next'
                if ('agents', 'reward') in eval_tensordict['next'].keys(include_nested=True):
                     step_reward = eval_tensordict['next'][('agents', 'reward')].sum().item() # Sum rewards across all agents and get scalar
                     total_episode_reward += step_reward
                else:
                     print("Warning: Agent-wise reward ('agents', 'reward') not found in the stepped tensordict.")


                # Update the done flag for the loop condition
                done = eval_tensordict['next']['done'] # Get done flag for the *next* state

            # Episode finished
            episode_rewards.append(total_episode_reward)
            print(f"  Episode {episode + 1} finished. Total reward: {total_episode_reward:.2f}")

        # Evaluation finished
        mean_eval_reward = sum(episode_rewards) / num_eval_episodes
        print(f"\nEvaluation complete over {num_eval_episodes} episodes.")
        print(f"Mean episode reward: {mean_eval_reward:.2f}")

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")

else:
    print("\nEvaluation environment or policy not available. Skipping evaluation loop.")
