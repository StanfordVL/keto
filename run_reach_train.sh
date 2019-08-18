python tools/run_env.py \
  --env ReachPointCloudEnv \
  --env_config configs/envs/reach_point_cloud_env.yaml \
  --is_training 1 \
  --policy ReachPointCloudPolicy \
  --policy_config configs/policies/reach_point_cloud_policy.yaml \
  --problem ReachPointCloudProblem \
  --episodic 0 --num_episodes 4096 --debug 1 \
  --output episodes/reach_point_cloud \
  --checkpoint keypoints/save/multiple_objects/reach/cvae_reach
