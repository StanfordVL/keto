# KeyPoint-based Task-Oriented Grasping

### Requirements

Follow the [instruction](https://docs.google.com/document/d/1y4F1_8G2u49ohP94eWUmlJeF7_b9IXk3IaYxg8aBJBc/edit?usp=sharing) to install the package.

Install `pybullet` following the [pybullet quickstart guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.opnfwdk9g3m).

### Usage

Run 4-dof grasping with random antipodal grasps:

```Shell
python tools/run_env.py \
         --env Grasp4DofEnv \
         --env_config configs/envs/grasp_4dof_env.yaml \
         --policy Grasp4DofRandomPolicy \
         --policy_config configs/policies/grasp_4dof_random_policy.yaml \
         --problem Grasp4DofProblem
         --episodic 0 \
         --num_episodes 100 \
         --debug 0 \
         --output episodes/grasp_4dof_random
```

Train Grasp-Quality Convolutional Neural Network (GQCNN) on Dex-Net dataset:
```Shell
python tools/batch_train.py \
         --env Grasp4DofEnv \
         --network GQCNN \
         --problem Grasp4DofProblem \
         --data_dir episodes/grasp_4dof_random \
         --working_dir outputs/gqcnn
```

Run 4-dof grasping with GQCNN:

```Shell
python tools/run_env.py \
         --env Grasp4DofEnv \
         --policy Grasp4DofCemPolicy \
         --policy_config configs/policies/grasp_4dof_cem_policy.yaml \
         --checkpoint outputs/gqcnn \
         --num_episodes 100 \
         --debug 1 \
         --output episodes/grasp_4dof_cem
```
