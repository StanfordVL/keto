python tools/run_env.py \
  --env HammerPointCloudEnv \
  --env_config configs/envs/hammer_point_cloud_env.yaml \
  --policy HammerPointCloudPolicy \
  --policy_config configs/policies/hammer_point_cloud_policy.yaml \
  --problem HammerPointCloudProblem \
  --episodic 0 --num_episodes 512 --debug 1 \
  --output episodes/hammer_point_cloud \
  --checkpoint runs/cvae_model