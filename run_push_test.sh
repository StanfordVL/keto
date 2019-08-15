python tools/run_env.py \
  --env PushPointCloudEnv \
  --env_config configs/envs/push_point_cloud_env.yaml \
  --is_training 0 \
  --policy PushPointCloudPolicy \
  --policy_config configs/policies/push_point_cloud_policy.yaml \
  --problem PushPointCloudProblem \
  --episodic 0 --num_episodes 8192 --debug 1  \
  --output episodes/push_point_cloud \
  --checkpoint keypoints/save/push/cvae_push