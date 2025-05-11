#!/bin/bash

source_path="/home/madjel/Projects/gitpackages/2D-to-3D pose lifter/smpl/amass/smplh/full_amass_3D_poses/"
target_path="laptop:/home/aladinedev/Projects/gitpackages/lifter-rsync/full_amass/"

rsync -azh "$source_path" "$target_path" \
      --progress \
      --exclude=".git" \
