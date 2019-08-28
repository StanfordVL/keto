python tools/run_env.py \
  --env CombinePointCloudEnv \
  --env_config configs/envs/combine_point_cloud_env.yaml \
  --is_training 0 \
  --policy CombinePointCloudPolicy \
  --policy_config configs/policies/combine_point_cloud_policy.yaml \
  --problem CombinePointCloudProblem \
  --episodic 0 --num_episodes 4096 --debug 1 \
  --output episodes/combine_point_cloud
