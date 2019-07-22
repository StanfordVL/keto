python tools/run_env.py \
  --env Grasp4DofEnv \
  --env_config configs/envs/grasp_4dof_env.yaml \
  --policy Grasp4DofRandomPolicy \
  --policy_config configs/policies/grasp_4dof_random_policy.yaml \
  --problem Grasp4DofProblem \
  --episodic 0 --num_episodes 4096 --debug 0 \
  --output episodes/grasp_4dof_random_single