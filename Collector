# @title  Collector
from torchrl.collectors import SyncDataCollector

collector = None
if env is not None and policy_network is not None:
    print("\nInstantiating SyncDataCollector...")
    try:
        frames_per_batch = env.num_envs # Collect one step from each environment per batch
        total_frames = 10000 # Example: collect a large number of frames for training

        collector = SyncDataCollector(
            env,
            policy=policy_network, # Use the instantiated policy network
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
            storing_device=device,
        )

        print("\nSyncDataCollector instantiated successfully.")
        print(f"Collector total frames: {collector.total_frames}")
        print(f"Collector frames per batch: {collector.frames_per_batch}")

    except Exception as e:
        print(f"\nAn error occurred during SyncDataCollector instantiation: {e}")
else:
    print("\nEnvironment or policy network not available. Cannot instantiate data collector.")
