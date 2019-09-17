import json

dict = {
  "part": {
    "obj_path": "./cvxparts/000000.obj",
    "translation": [0, 0, 0],
    "rotation": [0, 0, 0], 
    "config": {
        "mass_range": [0.2, 0.4],
        "size_range": [[0.02, 0.04], [0.02, 0.04], [0.20, 0.30]],
        "lateral_friction_range": [0.9, 1.0],
        "spinning_friction_range": [0.9, 1.0],
        "inertia_friction_range": [0.9, 1.0]
        }
  }
}

string = json.dumps(dict, indent=4, separators=(',', ': '))

with open('test.json', 'w') as f:
    f.writelines(string)
