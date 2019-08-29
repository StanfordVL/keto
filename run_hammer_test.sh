python tools/run_env.py \
  --env HammerPointCloudEnv \
  --env_config configs/envs/hammer_easy_env.yaml \
  --is_training 0 \
  --policy HammerPointCloudPolicy \
  --policy_config configs/policies/hammer_point_cloud_policy.yaml \
  --problem HammerPointCloudProblem \
  --episodic 0 --num_episodes 1024 --debug 1 \
  --output episodes/hammer_point_cloud \
  --checkpoint keypoints/keypoint_models/cvae_hammer
