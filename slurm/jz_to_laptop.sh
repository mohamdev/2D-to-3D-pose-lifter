#!/bin/bash

source_path="jz:/lustre/fswork/projects/rech/bwd/ugc42dc/pose-lifter/"
target_path="/home/aladinedev/Projects/gitpackages/lifter-rsync/"

rsync -azh "$source_path" "$target_path" \
      --progress \
      --exclude=".git" \
      --exclude="./smpl/amass/*"
