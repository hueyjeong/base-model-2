#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Setup SSH public key if provided
if [ -n "$SSH_PUB_KEY" ]; then
    echo "Setting up SSH public key from SSH_PUB_KEY..."
    mkdir -p /root/.ssh
    echo "$SSH_PUB_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi

# Decode Rclone token if provided
if [ -n "$RCLONE_CONFIG_GDRIVE_TOKEN_BASE64" ]; then
    echo "Decoding Rclone token from RCLONE_CONFIG_GDRIVE_TOKEN_BASE64..."
    export RCLONE_CONFIG_GDRIVE_TOKEN=$(echo "$RCLONE_CONFIG_GDRIVE_TOKEN_BASE64" | base64 -d)
fi

# Execute the CMD passed to the Docker container
exec "$@"
