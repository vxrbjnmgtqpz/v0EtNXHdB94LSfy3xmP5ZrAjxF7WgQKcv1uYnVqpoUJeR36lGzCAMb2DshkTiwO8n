#!/bin/bash

# 🚀 JAMNet Collaboration Setup Script
# Automated setup for new contributors and development environments

set -e  # Exit on any error

echo "🚀 JAMNet Collaboration Setup"
echo "============================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if we're in the right directory
if [[ ! -d "JAM_Framework_v2" ]] || [[ ! -d "TOASTer" ]]; then
    print_error "This script must be run from the JAMNet project root directory"
    exit 1
fi

print_info "Checking project structure..."
print_status "Found JAMNet project structure"

# Check Git configuration
echo ""
echo "📝 Git Configuration"
echo "===================="

if git config user.name >/dev/null 2>&1; then
    print_status "Git user.name: $(git config user.name)"
else
    print_warning "Git user.name not set"
    read -p "Enter your name: " user_name
    git config --local user.name "$user_name"
    print_status "Set git user.name to: $user_name"
fi

if git config user.email >/dev/null 2>&1; then
    print_status "Git user.email: $(git config user.email)"
else
    print_warning "Git user.email not set"
    read -p "Enter your email: " user_email
    git config --local user.email "$user_email"
    print_status "Set git user.email to: $user_email"
fi

# Configure collaboration settings
print_info "Configuring collaboration settings..."
git config --local pull.rebase true
git config --local rerere.enabled true
git config --local core.autocrlf input
print_status "Git collaboration settings configured"

# Check remotes
echo ""
echo "🔗 Remote Configuration"
echo "======================="

print_info "Current remotes:"
git remote -v

# Check if upstream is configured
if ! git remote | grep -q "upstream"; then
    print_warning "Upstream remote not configured"
    print_info "Adding upstream remote..."
    git remote add upstream https://github.com/vxrbjnmgtqpz/v0EtNXHdB94LSfy3xmP5ZrAjxF7WgQKcv1uYnVqpoUJeR36lGzCAMb2DshkTiwO8n.git
    print_status "Upstream remote added"
else
    print_status "Upstream remote already configured"
fi

# Sync with upstream
echo ""
echo "🔄 Syncing with Upstream"
echo "========================"

print_info "Fetching latest changes from upstream..."
git fetch upstream

if git rev-parse --verify upstream/main >/dev/null 2>&1; then
    print_info "Checking if local main is behind upstream..."
    LOCAL=$(git rev-parse main)
    UPSTREAM=$(git rev-parse upstream/main)
    
    if [ "$LOCAL" != "$UPSTREAM" ]; then
        print_warning "Local main is behind upstream, updating..."
        git checkout main
        git merge upstream/main
        git push origin main
        print_status "Local main updated and pushed to origin"
    else
        print_status "Local main is up to date with upstream"
    fi
else
    print_info "Upstream main not found, using origin"
fi

# Check development tools
echo ""
echo "🛠️ Development Tools Check"
echo "=========================="

# Check for cmake
if command -v cmake >/dev/null 2>&1; then
    print_status "CMake found: $(cmake --version | head -n1)"
else
    print_error "CMake not found"
    print_info "Install with: brew install cmake (macOS) or apt-get install cmake (Linux)"
fi

# Check for make
if command -v make >/dev/null 2>&1; then
    print_status "Make found: $(make --version | head -n1)"
else
    print_error "Make not found"
    print_info "Install with: brew install make (macOS) or apt-get install build-essential (Linux)"
fi

# Check for python
if command -v python3 >/dev/null 2>&1; then
    print_status "Python3 found: $(python3 --version)"
else
    print_error "Python3 not found"
    print_info "Install with: brew install python@3.11 (macOS) or apt-get install python3 (Linux)"
fi

# Test build
echo ""
echo "🔨 Testing Build System"
echo "======================="

print_info "Testing JAM Framework v2 build..."
cd JAM_Framework_v2

if [[ ! -d "build" ]]; then
    mkdir build
    print_info "Created build directory"
fi

cd build

if cmake .. >/dev/null 2>&1; then
    print_status "CMake configuration successful"
    
    if make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) >/dev/null 2>&1; then
        print_status "Build test successful"
    else
        print_warning "Build test failed - may need dependencies"
    fi
else
    print_warning "CMake configuration failed - may need dependencies"
fi

cd ../..

# Create sample branch for testing
echo ""
echo "🌿 Branch Setup"
echo "==============="

current_branch=$(git branch --show-current)
print_info "Current branch: $current_branch"

if [[ "$current_branch" == "main" ]]; then
    branch_name="feature/setup-test-$(date +%Y%m%d-%H%M%S)"
    print_info "Creating test feature branch: $branch_name"
    git checkout -b "$branch_name"
    print_status "Created and switched to branch: $branch_name"
    
    # Create a simple test file
    echo "# Setup Test - $(date)" > SETUP_TEST.md
    echo "This file was created during collaboration setup." >> SETUP_TEST.md
    echo "User: $(git config user.name)" >> SETUP_TEST.md
    echo "Email: $(git config user.email)" >> SETUP_TEST.md
    echo "Branch: $branch_name" >> SETUP_TEST.md
    
    git add SETUP_TEST.md
    git commit -m "test: Add setup verification file"
    print_status "Created test commit"
    
    print_info "You can now push this branch with: git push -u origin $branch_name"
fi

# GPU/System Info
echo ""
echo "🖥️ System Information"
echo "===================="

print_info "System: $(uname -s) $(uname -r)"
print_info "Architecture: $(uname -m)"

# Check for GPU info (macOS)
if command -v system_profiler >/dev/null 2>&1; then
    gpu_info=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -n1 | cut -d: -f2 | xargs)
    if [[ -n "$gpu_info" ]]; then
        print_status "GPU: $gpu_info"
    fi
fi

# Check for CUDA (if available)
if command -v nvcc >/dev/null 2>&1; then
    print_status "CUDA found: $(nvcc --version | grep "release" | cut -d, -f2 | xargs)"
else
    print_info "CUDA not found (normal for non-NVIDIA systems)"
fi

# Final instructions
echo ""
echo "🎯 Next Steps"
echo "============="
echo ""
print_info "1. Review the project structure:"
echo "   - JAM_Framework_v2/: Core UDP/GPU-native framework"
echo "   - TOASTer/: Transport layer protocols"
echo "   - JDAT_Framework/: Data analysis tools"
echo "   - Documentation/: Project documentation"
echo ""
print_info "2. Key concepts to understand:"
echo "   - This is GPU-DOMINANT, not GPU-accelerated"
echo "   - CPU serves the GPU, not the other way around"
echo "   - Pure UDP networking (no TCP/HTTP)"
echo "   - JSONL streaming format"
echo ""
print_info "3. Start with documentation:"
echo "   - Read README files in each framework"
echo "   - Check Documentation/ folder"
echo "   - Run existing tests to understand the system"
echo ""
print_info "4. Development workflow:"
echo "   - Always sync with upstream before starting work"
echo "   - Create feature branches for all changes"
echo "   - Test your changes before creating PRs"
echo ""

if [[ "$current_branch" != "main" ]]; then
    print_status "Setup complete! You're ready to contribute to JAMNet."
    print_info "Current working branch: $current_branch"
else
    print_status "Setup complete! Created test branch for you to experiment with."
    print_info "Switch back to main with: git checkout main"
fi

echo ""
print_status "JAMNet collaboration setup finished successfully! 🚀"
