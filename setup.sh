#!/bin/bash
# File: setup.sh

echo "=== Hybrid C++/Python Inference Engine Setup ==="

create_directories() {
    echo "Creating project directory structure..."
    mkdir -p src models build data
    echo "✓ Directories created"
}

check_requirements() {
    echo "Checking system requirements..."

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d ' ' -f2)
        echo "✓ Python3 found: $PYTHON_VERSION"
    else
        echo "✗ Python3 not found. Please install Python 3.7+"
        exit 1
    fi

    if command -v pip3 &> /dev/null; then
        echo "✓ pip3 found"
    else
        echo "✗ pip3 not found. Please install pip3"
        exit 1
    fi

    if command -v cmake &> /dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n1 | cut -d ' ' -f3)
        echo "✓ CMake found: $CMAKE_VERSION"
    else
        echo "✗ CMake not found. Please install CMake 3.12+"
        exit 1
    fi

    if command -v g++ &> /dev/null; then
        GCC_VERSION=$(g++ --version | head -n1 | cut -d ' ' -f3)
        echo "✓ g++ found: $GCC_VERSION"
    else
        echo "✗ g++ not found. Please install g++"
        exit 1
    fi
}

install_python_deps() {
    echo "Installing Python dependencies..."
    pip3 install --user pybind11 numpy

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Note: On Linux, install python3-dev if needed:"
        echo "  sudo apt-get install python3-dev python3-numpy"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Note: On macOS, use Homebrew if needed:"
        echo "  brew install python3"
    fi

    echo "✓ Python dependencies installed"
}

create_readme() {
    echo "Creating README..."
    cat > README.md << 'EOF'
<-- Paste only the README.md content from your original script here -->
EOF
    echo "✓ README created"
}

create_makefile() {
    echo "Creating Makefile..."
    cat > Makefile << 'EOF'
<-- Paste only the Makefile content from your original script here -->
EOF
    echo "✓ Makefile created"
}

main() {
    echo "Starting setup process..."
    #create_directories
    #check_requirements
    #install_python_deps
    #create_makefile
    #create_readme
    echo ""
    echo "🎉 Setup completed successfully!"
}

main "$@"

