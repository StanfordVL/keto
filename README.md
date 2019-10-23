## KETO: Learning Keypoint Representations for Tool Manipulation

### Install

Clone this repository
```bash
git clone https://github.com/XXX/keto.git
```

Download and extract [data.tar.gz] in `keto` and [models.tar.gz] in `keto/keypoints`. Then the directory should contain `keto/data` and `keto/keypoints/models`. Create a clean virtual environment with Python 3.6. If you are using Anaconda, you could run `conda create -n keto python=3.6` to create an environment named keto and activate this environment by `conda activate keto`. 

Install the dependencies:
```bash
python setup.py
```

### Quick Demo
Run pushing:
```bash
sh scripts/run_push_test.sh
```

Run reaching:
```bash
sh scripts/run_reach_test.sh
```

Run hammering:
```bash
sh scripts/run_hammer_test.sh
```

### Train from Scratch

#### Grasping
Run the random grasping policy to collect training data:
```bash
sh scripts/run_grasp_random.sh
```
In the experiment, we ran 300 copies of `run_grasp_random.sh` in parallel on machines with over 300 cpu cores, collecting 100K episodes. The data will be saved to `episodes/grasp_4dof_random/grasp`, including the grasp location, rotation and a binary value indicating whether the grasp succeeded. `episodes/grasp_4dof_random/point_cloud` stores the input point cloud (visual observation) associated with each grasp. Our next step is to train a neural network that can generate successful grasps from the input point cloud, which is expected to achieve a higher successful rate than the random policy.

First, we store the collected data in a single hdf5 file to boost the data access in training:
```bash
cd keypoints && mkdir data && python utils/grasp/make_inputs_multiproc.py --point_cloud ../episodes/grasp_4dof_random/point_cloud --grasp ../episodes/grasp_4dof_random/grasp --save data/data_grasp.hdf5
```

