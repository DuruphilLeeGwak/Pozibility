#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

hallu="$1"

cd .. #project
cd data/input
img_name=$(ls -td * | head -n 1)
echo "img path: $img_name"
cd .. #data
cd .. #project

cp data/qwen_outputs/qwen_output.jpg modules/NBGenerator/input
cp data/input/"$img_name" modules/NBGenerator/input

latest_dir=$(ls -td modules/PoseExtractor/outputs/* | head -n 1)
echo "latest output: $latest_dir"
pose_path="$latest_dir/ref/ref_rend.jpg"
cp "$pose_path" modules/NBGenerator/input

cd modules/NBGenerator
conda activate pose_env


python main.py --qwen_result input/qwen_output.jpg --img_original input/"$img_name" --img_rendered input/ref_rend.jpg --hallucination_type "$hallu"
conda deactivate

cd .. #modules
cd .. #project

output_path=$(ls -td modules/NBGenerator/output/* | head -n 1)
echo "latest output: $output_path"
mv "$output_path" data/outputs