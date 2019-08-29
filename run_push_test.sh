python tools/run_env.py \
  --env PushPointCloudEnv \
  --env_config configs/envs/push_easy_env.yaml \
  --is_training 0 \
  --policy PushPointCloudPolicy \
  --policy_config configs/policies/push_point_cloud_policy.yaml \
  --problem PushPointCloudProblem \
  --episodic 0 --num_episodes 8192 --debug 1  \
  --output episodes/push_point_cloud \
  --checkpoint keypoints/keypoint_models/cvae_push
