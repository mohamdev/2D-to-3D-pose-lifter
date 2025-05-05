#!/bin/bash

source_path="/home/aladinedev/Projects/gitpackages/lifter-rsync/"
target_path="jz:/lustre/fswork/projects/rech/bwd/ugc42dc/pose-lifter/"

# FIRST SYNC RIGHT AWAY
rsync -azh "$source_path" "$target_path" \
      --progress \
      --delete --force \
      --exclude=".git" \
      --exclude="./smpl/amass/smplx/" \
      --exclude="./smpl/amass/smplh/joint_segment_data/" \
      --exclude="./smpl/amass/smplh/joint_segment_data_old/" \
      --exclude="./smpl/amass/smplh/latent_npz_data/" \
      --exclude="./smpl/amass/smplh/raw_npz_data/"
      
while inotifywait -r -e modify,create,delete "$source_path"
do
    rsync -azh "$source_path" "$target_path" \
          --progress \
          --delete --force \
          --exclude=".git" \
          --exclude="./smpl/amass/smplx/" \
          --exclude="./smpl/amass/smplh/joint_segment_data/" \
          --exclude="./smpl/amass/smplh/joint_segment_data_old/" \
          --exclude="./smpl/amass/smplh/latent_npz_data/" \
          --exclude="./smpl/amass/smplh/raw_npz_data/" 
done