#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

cd ..
cd data/input
img_name=$(ls -td * | head -n 1)
echo "img path: $img_name"
pose_name=$(ls -td * | head -n 2 | tail -n 1)
echo "pose path: $pose_name"
cd ..
cd ..

#========Pose Extractor========
cp data/input/"$img_name" modules/PoseExtractor/inputs/src
cp data/input/"$pose_name" modules/PoseExtractor/inputs/ref

cd modules/PoseExtractor
conda activate pose_env
python PoseExtractor.py --source inputs/src/"$img_name" --reference inputs/ref/"$pose_name"
latest_dir=$(ls -td outputs/* | head -n 1)
echo "latest output: $latest_dir"

cd .. #/modules
pose_path="PoseExtractor/$latest_dir/trans/trans_sk.jpg"
echo "$pose_path"
cp "$pose_path" qwen/ComfyUI/input
#==============================

#=======Image Captioning=======
mv PoseExtractor/inputs/src/"$img_name" ImageCaptioning/input
cd ImageCaptioning
python main.py --img_path input/"$img_name" --type img_org

cd .. #modules
mv ImageCaptioning/prompt.json qwen/prompt
mv ImageCaptioning/input/"$img_name" qwen/ComfyUI/input

conda deactivate
#==============================


#=============QWEN=============
cd qwen
conda activate qwen
python main.py --img_path "$img_name" --skeleton_path trans_sk.jpg --prompt prompt/prompt.json
conda deactivate

cd .. #modules
cd .. #project
cp modules/qwen/output/qwen_output.jpg data/qwen_outputs
#==============================