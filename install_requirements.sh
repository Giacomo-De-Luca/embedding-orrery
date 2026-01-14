#!/bin/bash

# Exit on error
set -e

echo "Starting project setup..."

# --- 1. Python (uv) Setup ---
echo ""
echo "Setting up Python environment with uv..."

if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the environment if needed (uv usually adds to path, but we might need to source cargo env or similar if it was just installed)
    # For now, we assume the user might need to restart shell or we use the full path if we knew it, 
    # but usually the installer updates the path for the next session. 
    # We can try to source the cargo env which uv might use if installed via cargo, but the curl script installs to ~/.cargo/bin or ~/.local/bin
    
    # Try to add common install locations to PATH for this script execution
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
else
    echo "uv is already installed."
fi

echo "Installing Python dependencies..."
uv sync

# --- 2. Rust Setup ---
echo ""
echo "🦀 Setting up Rust environment..."
if [ -f "./setup_rust.sh" ]; then
    chmod +x ./setup_rust.sh
    ./setup_rust.sh
else
    echo "Error: setup_rust.sh not found!"
    exit 1
fi

# --- 3. Node.js/npm Setup ---
echo ""
echo "Setting up Node.js environment..."

if ! command -v node &> /dev/null || ! command -v npm &> /dev/null; then
    echo "Node.js/npm not found. Installing via nvm..."
    
    # Install nvm if not present
    if [ ! -d "$HOME/.nvm" ]; then
        echo "Installing nvm..."
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    fi
    
    # Load nvm for this script session
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    
    # Install latest LTS version of Node.js
    echo "Installing Node.js LTS..."
    nvm install --lts
    nvm use --lts
    
    # Verify installation
    if ! command -v node &> /dev/null || ! command -v npm &> /dev/null; then
        echo "Error: Failed to install Node.js/npm. Please install manually."
        exit 1
    fi
    
    echo "Node.js $(node --version) and npm $(npm --version) installed successfully!"
else
    echo "Node.js $(node --version) and npm $(npm --version) are already installed."
fi

# --- 4. JavaScript Dependencies ---
echo ""
echo "Installing JavaScript dependencies..."
if [ -d "embedding_visualization" ]; then
    cd embedding_visualization
    
    echo "Installing npm dependencies..."
    npm install
    
    cd ..
else
    echo "Error: embedding_visualization directory not found!"
    exit 1
fi

echo ""
echo "All requirements installed successfully!"
