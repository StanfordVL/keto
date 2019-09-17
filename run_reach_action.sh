python tools/run_env.py \
  --env ReachPointCloudEnv \
  --env_config configs/envs/reach_hard_env.yaml \
  --is_training 0 \
  --policy ReachActionPolicy \
  --policy_config configs/policies/reach_point_cloud_policy.yaml \
  --problem ReachPointCloudProblem \
  --episodic 0 --num_episodes 4096 --debug 1 \
  --output episodes/reach_point_cloud \
  --checkpoint keypoints/action_models/cvae_action_reach
