#!/bin/sh

prefix="$(pwd)/Poles"


for modality in new_rgb new_lidar; do
cat > "$prefix/$modality/data.yaml" <<EOF
train: $prefix/$modality/images/train
val: $prefix/$modality/images/valid
test: $prefix/$modality/images/test

nc: 1
names: ['pole']

nc: 1
names: ['pole']
EOF
done

