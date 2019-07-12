# RoboVat

<p align="center"><img width="80%" src="docs/brains-in-a-vat.gif" /></p>

An object-oriented robot learning framework for both the simulation and the real world.

### Requirements:

Follow the [instruction](https://docs.google.com/document/d/1y4F1_8G2u49ohP94eWUmlJeF7_b9IXk3IaYxg8aBJBc/edit?usp=sharing) to install the package.

Install `pybullet` following the [pybullet quickstart guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.opnfwdk9g3m).

### Command Line Interface:
Run the command line interface in simulation or the real world:

```Shell
python tools/run_env.py \
         --env grasp_4dof \
         --policy Grasp4DofRandomPolicy \
         --policy_config configs/policies/grasp_4dof_random_policy.yaml \
         --num_episodes 100 \
         --debug 1
```
