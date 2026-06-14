#!/bin/bash

# bash script to convert CHAOS dcm images into nifti files (T1DUAL + T2SPIR)
# usage: $0 [input_dir] [output_dir]

if [ "$#" -ne 2 ]; then
    echo "usage: $0 [input] [output]"
    exit 1
fi

INPUT_DIR=$1
BASE_OUTPUT_DIR=$2

for sid in "$INPUT_DIR"/*; do
    if [ -d "$sid" ]; then
        subject_name=$(basename "$sid")
        for modality in T1DUAL T2SPIR; do
            dicom_dir="$sid/$modality/DICOM_anon"
            OUTPUT_DIR="$BASE_OUTPUT_DIR/$modality"
            JSON_DIR="$BASE_OUTPUT_DIR/json/$modality"
            mkdir -p "$OUTPUT_DIR" "$JSON_DIR"
            if [ -d "$dicom_dir" ]; then
                echo "Convert subject $subject_name $modality..."
                dcm2niix -o "$OUTPUT_DIR" -f "chaos_${subject_name}_${modality}" -z y "$dicom_dir"
                mv "$OUTPUT_DIR"/*.json "$JSON_DIR"/ 2>/dev/null || true
            fi
        done
    fi
done

echo "Conversion completed"