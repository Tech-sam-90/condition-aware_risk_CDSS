#!/bin/bash
# download_icu_files.sh
# Bash script to download selected ICU files from PhysioNet for CDSS project
# Downloads only once asking for password a single time

# Change to your raw data directory
cd ~/condition-aware_risk_CDSS/data/raw || exit

# PhysioNet credentials
PHYSIONET_USER="sadeniji"

# Files to download
FILES=(
    "hosp/d_icd_diagnoses.csv.gz"
)

# Loop over files and download each
for FILE in "${FILES[@]}"; do
    echo "Downloading $FILE ..."
    wget -c --user "$PHYSIONET_USER" --ask-password "https://physionet.org/files/mimiciv/3.1/$FILE"
done

echo "All ICU files downloaded!"

