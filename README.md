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
Run pushing
```bash
sh scripts/run_push_test.sh
```

Run reaching
```bash
sh scripts/run_reach_test.sh
```

Run hammering
```bash
sh scripts/run_hammer_test.sh
```

### Train from Scratch

#### Grasping
Run the random grasping policy to collect training data
```bash
sh scripts/run_grasp_random.sh
```
The data will be saved to `episodes/grasp_4dof_random`, including the grasp location, rotation and a binary value indicating whether the grasp succeeded. In the experiment, we ran 100K episodes, which could take 300 hours if there was only a single process. In practice, we would suggest to run `run_grasp_random.sh` multiple times in parallel on machines with abundant cpu cores.
