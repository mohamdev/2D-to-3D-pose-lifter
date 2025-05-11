#!/bin/bash

source_path="laptop:/home/aladinedev/Projects/gitpackages/lifter-rsync/trained_models/"
target_path="/home/madjel/Projects/gitpackages/2D-to-3D pose lifter/trained_models/"

rsync -azh "$source_path" "$target_path" \
      --progress \
      --exclude=".git" \
      --exclude="./smpl/amass/"