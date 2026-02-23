#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Execute the CMD passed to the Docker container
exec "$@"
