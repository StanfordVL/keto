import os

# Install the dependencies
os.system('pip install --upgrade numpy scipy h5py pyyaml future opencv-python matplotlib easydict gym sklearn python-pcl cvxpy')

os.system('pip install pillow --no-cache-dir')

# Install Tensorflow
os.system('pip install tf-nightly-gpu==1.13.0.dev20190117')

os.system('pip install tf-agents==0.2.0rc2')

os.system('pip install tensorflow-probability==0.5.0')

os.system('pip install tf-estimator-nightly==1.13.0.dev2019010910')

# Install Pybullet
os.system('pip install -e git+https://github.com/bulletphysics/bullet3@6a74f63604ceecd1db5c71036ffb0dbf17294579#egg=pybullet')
