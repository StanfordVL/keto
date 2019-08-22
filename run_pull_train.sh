python tools/run_env.py \
  --env PullPointCloudEnv \
  --env_config configs/envs/pull_point_cloud_env.yaml \
  --is_training 1 \
  --policy PullPointCloudPolicy \
  --policy_config configs/policies/pull_point_cloud_policy.yaml \
  --problem PullPointCloudProblem \
  --episodic 0 --num_episodes 8192 --debug 1  \
  --output episodes/pull_point_cloud \
  --checkpoint keypoints/save/multiple_objects/pull/cvae_pull
