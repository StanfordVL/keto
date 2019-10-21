python tools/run_env.py \
  --env HammerPointCloudEnv \
  --env_config configs/envs/hammer_env.yaml \
  --is_training 1 \
  --policy HammerPointCloudPolicy \
  --policy_config configs/policies/hammer_point_cloud_policy.yaml \
  --problem HammerPointCloudProblem \
  --episodic 0 --num_episodes 200000 --debug 1 \
  --output episodes/hammer_point_cloud \
  --checkpoint keypoints/models/cvae_hammer
