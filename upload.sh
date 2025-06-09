#!/bin/bash

# Check if at least one input is provided
if [ $# -eq 0 ]; then
    echo "Error: No input provided"
    echo "Usage: bash upload.sh file1.py [file2.py dir1 dir2 ...]"
    exit 1
fi

# Store the current directory
CURRENT_DIR=$(pwd)

# Arrays to track files and directories
FILES=()
DIRS=()

# Check if inputs exist and categorize them
for INPUT in "$@"; do
    if [ -f "$INPUT" ]; then
        FILES+=("$INPUT")
    elif [ -d "$INPUT" ]; then
        DIRS+=("$INPUT")
    else
        # Check if it might be intended as a directory but doesn't exist
        if [[ "$INPUT" != *"."* ]]; then
            echo "Error: Directory $INPUT does not exist"
        else
            echo "Error: File $INPUT does not exist"
        fi
        exit 1
    fi
done

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Clone the repository
echo "Cloning repository..."
git clone https://github.com/Kishor-Bhaumik/Data-Mining235.git
if [ $? -ne 0 ]; then
    echo "Error: Failed to clone repository"
    cd "$CURRENT_DIR"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Copy files to the repository directory
echo "Copying files and directories to repository..."

# First handle files
for FILE in "${FILES[@]}"; do
    cp "$CURRENT_DIR/$FILE" Data-Mining235/
    echo "Copied file: $FILE"
done

# Then handle directories
for DIR in "${DIRS[@]}"; do
    DIR_NAME=$(basename "$DIR")
    mkdir -p "Data-Mining235/$DIR_NAME"
    cp -r "$CURRENT_DIR/$DIR"/* "Data-Mining235/$DIR_NAME"/
    echo "Copied directory: $DIR"
done

# Change to the repository directory
cd Data-Mining235 

# Add all files and directories to git
echo "Adding files and directories to git..."
git add .

# Commit the changes
echo "Committing changes..."
if [ $# -eq 1 ]; then
    # Single item upload
    if [ ${#FILES[@]} -eq 1 ]; then
        COMMIT_MSG="Add file: ${FILES[0]}"
    else
        COMMIT_MSG="Add directory: ${DIRS[0]}"
    fi
else
    # Multiple items upload
    COMMIT_MSG="Add multiple items:"
    for FILE in "${FILES[@]}"; do
        COMMIT_MSG="$COMMIT_MSG file:$FILE,"
    done
    for DIR in "${DIRS[@]}"; do
        COMMIT_MSG="$COMMIT_MSG dir:$DIR,"
    done
    # Remove trailing comma
    COMMIT_MSG=${COMMIT_MSG%,}
fi

git commit -m "$COMMIT_MSG"
if [ $? -ne 0 ]; then
    echo "Error: Failed to commit changes"
    cd "$CURRENT_DIR"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Push the changes to GitHub
echo "Pushing changes to GitHub..."
git push origin main
if [ $? -ne 0 ]; then
    echo "Error: Failed to push changes"
    cd "$CURRENT_DIR"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Return to the original directory
cd "$CURRENT_DIR"

# Clean up the temporary directory
rm -rf "$TEMP_DIR"

echo "Successfully uploaded to GitHub repository:"
for FILE in "${FILES[@]}"; do
    echo "- File: $FILE"
done
for DIR in "${DIRS[@]}"; do
    echo "- Directory: $DIR"
done