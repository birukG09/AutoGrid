#!/bin/bash
# AutoGrid Multi-Language Build Script

echo "🔧 Building AutoGrid Distributed Energy Controller..."
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Build C++ embedded controller
echo -e "${YELLOW}Building C++ Embedded Controller...${NC}"
if command -v cmake &> /dev/null; then
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    cd ..
    echo -e "${GREEN}✓ C++ controller built successfully${NC}"
else
    echo -e "${RED}⚠ CMake not found - C++ controller will use simulation mode${NC}"
fi

# Build Rust coordination layer
echo -e "${YELLOW}Building Rust Coordination Layer...${NC}"
if command -v cargo &> /dev/null; then
    cargo build --release
    echo -e "${GREEN}✓ Rust coordinator built successfully${NC}"
else
    echo -e "${RED}⚠ Cargo not found - Rust coordinator will use simulation mode${NC}"
fi

# Setup Python AI engine
echo -e "${YELLOW}Setting up Python AI Engine...${NC}"
if command -v python3 &> /dev/null; then
    python3 -c "import numpy, pandas" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Python AI engine dependencies available${NC}"
    else
        echo -e "${YELLOW}Installing Python dependencies...${NC}"
        pip3 install numpy pandas scikit-learn --quiet
    fi
else
    echo -e "${RED}⚠ Python3 not found${NC}"
fi

# Set executable permissions
chmod +x main.py
chmod +x build.sh

echo ""
echo -e "${GREEN}🚀 AutoGrid build completed!${NC}"
echo ""
echo "Usage:"
echo "  ./main.py              - Start AutoGrid console"
echo "  ./build.sh             - Rebuild all components"
echo ""
echo "Architecture:"
echo "  ├── C++/ARM Embedded Controller (Real-time hardware control)"
echo "  ├── Rust Coordination Layer (Distributed consensus & safety)"
echo "  ├── Python AI Engine (Machine learning & forecasting)"
echo "  └── Console Interface (Green/White/Red themed terminal UI)"
echo ""