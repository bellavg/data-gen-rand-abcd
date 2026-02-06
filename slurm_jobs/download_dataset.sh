#!/bin/bash
#SBATCH -p staging            # Use the legal staging partition
#SBATCH -t 36:00:00           # 1.4TB download + unzip takes time
#SBATCH --job-name=openabc_init
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# 1. SETUP DIRECTORIES
TARGET_DIR="/scratch-shared/$USER/openabc_full"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# API Variables (NYU UltraViolet)
RECORD_ID="mw6q2-a8p15"
BASE_URL="https://ultraviolet.library.nyu.edu/api/records/$RECORD_ID/files"

# 2. DOWNLOAD SPLIT PARTS (z01 to z13)
for i in {01..13}; do
    FILE="OPENABC_DATASET.z$i"
    
    # Skip if file already exists and is ~100GB (107374182400 bytes)
    # Using 90GB as a safe threshold for "mostly complete"
    if [ -f "$FILE" ] && [ $(stat -c%s "$FILE") -gt 90000000000 ]; then
        echo ">> $FILE already exists and looks complete. Skipping to next..."
        continue
    fi

    echo ">> Downloading $FILE..."
    # Removed -C - because the server doesn't support byte ranges/resuming
    until curl -L -X GET "$BASE_URL/$FILE/content" -o "$FILE"; do
        echo "!! Connection dropped for $FILE. Deleting partial and restarting in 10 seconds..."
        rm -f "$FILE"
        sleep 10
    done
    echo ">> $FILE is complete."
done

# 3. DOWNLOAD THE MAIN ZIP INDEX
if [ ! -f "OPENABC_DATASET.zip" ]; then
    echo ">> Downloading OPENABC_DATASET.zip..."
    until curl -L -X GET "$BASE_URL/OPENABC_DATASET.zip/content" -o "OPENABC_DATASET.zip"; do
        echo "!! Zip download interrupted. Restarting in 10 seconds..."
        rm -f "OPENABC_DATASET.zip"
        sleep 10
    done
fi

echo ">> All parts downloaded successfully."

# 4. EXTRACT (Directly from the split archive)
# We do NOT use 'cat' here. Unzip is designed to handle split files 
# automatically if the .z01, .z02... files are in the same folder as the .zip
echo ">> Extracting specific folders (this saves space and inodes)..."
unzip -q OPENABC_DATASET.zip \
    "OPENABC_DATASET/bench/*" \
    "OPENABC_DATASET/statistics/*" \
    "OPENABC_DATASET/synScripts/*" \
    "OPENABC_DATASET/lib/*"

# 5. CLEANUP
# Only delete the installers once you are sure the extraction worked.
if [ $? -eq 0 ]; then
    echo ">> Extraction successful. Cleaning up installer files..."
    rm -f OPENABC_DATASET.z* OPENABC_DATASET.zip
else
    echo "!! Extraction failed. Keeping files for manual inspection."
    exit 1
fi

echo ">> Job Complete.