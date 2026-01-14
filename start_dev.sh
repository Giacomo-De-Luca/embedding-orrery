#!/bin/bash

# Ensure we are in the script's directory (project root)
cd "$(dirname "$0")"

# Open a new Terminal window/tab for the frontend
# We use osascript to automate the Terminal application
osascript -e 'tell application "Terminal" to do script "cd \"'$(pwd)'/embedding_visualization\" && npm run dev"'

# Run the backend in the current terminal
# This blocks until the backend is stopped
bash start_backend.sh
