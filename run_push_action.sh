python tools/run_env.py \
  --env PushPointCloudEnv \
  --env_config configs/envs/push_hard_env.yaml \
  --is_training 0 \
  --policy PushActionPolicy \
  --policy_config configs/policies/push_point_cloud_policy.yaml \
  --problem PushPointCloudProblem \
  --episodic 0 --num_episodes 8192 --debug 1  \
  --output episodes/push_point_cloud \
  --checkpoint keypoints/action_models/cvae_action_push
