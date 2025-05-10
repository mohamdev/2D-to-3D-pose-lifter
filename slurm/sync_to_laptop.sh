#!/bin/bash

source_path="/home/madjel/Projects/gitpackages/2D-to-3D pose lifter/"
target_path="laptop:/home/aladinedev/Projects/gitpackages/lifter-rsync/"

rsync -azh "$source_path" "$target_path" \
      --progress \
      --exclude=".git" \

# source_path="/home/madjel/Projects/gitpackages/2D-to-3D pose lifter/"
# target_path="laptop:/home/aladinedev/Projects/gitpackages/lifter-rsync/"

# while inotifywait -r -e modify,create,delete "$source_path"
# do
#     rsync -azh "$source_path" "$target_path" \
#           --progress \
#           --delete --force \
#           --exclude=".git" \
#           --exclude="./smpl/amass/smplx/" \
#           --exclude="./smpl/amass/smplh/joint_segment_data/" \
#           --exclude="./smpl/amass/smplh/joint_segment_data_old/" \
#           --exclude="./smpl/amass/smplh/latent_npz_data/" \
#           --exclude="./smpl/amass/smplh/raw_npz_data/" 
# done
