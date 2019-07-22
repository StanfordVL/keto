python tools/run_env.py \
  --env Grasp4DofPointCloudEnv \
  --env_config configs/envs/grasp_4dof_point_cloud_env.yaml \
  --policy Grasp4DofPointCloudPolicy \
  --policy_config configs/policies/grasp_4dof_point_cloud_policy.yaml \
  --problem Grasp4DofPointCloudProblem \
  --episodic 0 --num_episodes 4096 --debug 1 \
  --output episodes/grasp_4dof_point_cloud \
  --checkpoint runs/cvae_model
