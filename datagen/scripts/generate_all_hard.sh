#!/bin/bash

python generate.py --body t --output outputs/hard/T --num 5000 --color realistic --obj 'data/convex_mesh/*/*.obj' 
python generate.py --body x --output outputs/hard/X --num 5000 --color realistic --obj 'data/convex_mesh/*/*.obj' 
python generate.py --body l --output outputs/hard/L --num 5000 --color realistic --obj 'data/convex_mesh/*/*.obj' 
