# Procedural Object Generation

The procedural object generation used in [Learning Task-Oriented Grasping for Tool Manipulation from Simulated Self-Supervision](https://sites.google.com/view/task-oriented-grasp/).


### Citation
```
@inproceedings{fang2018rss,
author = {Kuan Fang, Yuke Zhu, Animesh Garg, Andrey Kuryenkov, Viraj Mehta, Li Fei-Fei, Silvio Savarese},
title = {Learning Task-Oriented Grasping for Tool Manipulation from Simulated Self-Supervision},
booktitle = {Robotics: Science and Systems (RSS)},
month = {June},
year = {2018}
}
```

### Prepare
1. Normalize .obj mesh files:

    ```Shell
    python normalize_mesh.py --input data/example/duck_vhacd.obj --output outputs/
    ```

2. Visualize a .urdf or .obj file:

    ```Shell
    python visualize.py --input outputs/000000/body.urdf
    ```
    or
    ```Shell
    python visualize.py --input outputs/000000/head.obj
    ```

### Generation
1. Generate realistic objects:

    ```Shell
    python generate.py --body t --color realistic --mesh data/example/duck_vhacd.obj --output outputs/ --num 1
    ```
