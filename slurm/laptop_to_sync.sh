#!/bin/bash

source_path="laptop:/home/aladinedev/Projects/gitpackages/lifter-rsync/"
target_path="/home/madjel/Projects/gitpackages/2D-to-3D pose lifter/"

rsync -azh "$source_path" "$target_path" \
      --progress \
      --exclude=".git" \
      --exclude="./smpl/amass/"