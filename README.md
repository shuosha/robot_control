# Robot Control

Robot control system for real robot environment setup and operation.

## Installation

### Prerequisites

- Ubuntu (tested on Ubuntu 20.04+)
- `uv` package manager ([installation guide](https://github.com/astral-sh/uv))

### Python Environment Setup with uv (Python 3.9)

1. **Create virtual environment with Python 3.9:**
   ```bash
   uv venv --python 3.9
   ```

2. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Install project dependencies:**
   ```bash
   uv sync
   ```

4. **Install PyTorch with CUDA support (for foundation pose):**
   ```bash
   uv pip uninstall setuptools && uv pip install setuptools==69.5.1
   uv pip install torchvision==0.16.0+cu121 torchaudio==2.1.0 torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Install PyTorch3D:**
   ```bash
   uv pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
   ```

## Hardware Setup

### Intel RealSense Camera

1. **Update Ubuntu:**
   ```bash
   sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade
   ```

2. **Install core packages:**
   ```bash
   sudo apt-get install libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev v4l-utils
   ```

3. **Install build tools:**
   ```bash
   sudo apt-get install git wget cmake build-essential
   ```

4. **Prepare Linux backend and development environment:**
   > **Note:** Unplug any connected RealSense cameras before proceeding.
   ```bash
   sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at
   ```

5. **Build and install librealsense:**
   ```bash
   cd src/third_party/librealsense
   git checkout v2.56.5
   ./scripts/setup_udev_rules.sh
   mkdir build && cd build
   cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=true -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=$(which python)
   cd ../../../
   sudo make uninstall && make clean && make -j8 && sudo make install
   ```

6. **Test RealSense viewer:**
   ```bash
   realsense-viewer
   ```

### Gello Robot Setup

1. **Install DYNAMIXEL Wizard 2.0:**
   - Download from: https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/
   - Change permissions:
     ```bash
     sudo chmod 775 DynamixelWizard2Setup_x64
     ```
   - Install:
     ```bash
     ./DynamixelWizard2Setup_x64
     ```
   - Follow the installation wizard (click next to proceed)

2. **Add user to dialout group:**
   ```bash
   sudo usermod -aG dialout <your_account_id>
   sudo reboot
   ```

3. **Clone and setup Gello:**
   ```bash
   python scripts/gello_get_offset.py --start-joints 0 -0.79 0 0.52 0 1.31 0 --joint-signs 1 1 1 1 1 1 1 --port /dev/ttyUSB0
   ```
   > **Note:** Verify port, baudrate, and robot IP settings as needed.
