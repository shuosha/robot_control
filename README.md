# Isaac simulation installation:

check GLIBC>= 2.35 by `ldd --version`

set up python env:
`uv venv --python 3.11 isaac`
`source isaac/bin/activate`

upgrade pip
`uv pip install --upgrade pip`

Install pytorch for CUDA 12.8:
`uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128`

Install isaacsim via pip:
`uv pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com`

# Real robot env installation
Install the following dependencies via pip
`uv pip install -r <(uv pip compile pyproject.toml)`

*Note: open3d requires numpy<2.0.0*

(optional) rerun torch installation to use torch==2.7.1+cu128

Install lerobot:
`cd third-party/lerobot`
`uv pip install -e .`

(TODO: add openpi)


# Realsense viewer
update ubuntu
`sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade`

install core packages
`sudo apt-get install libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev v4l-utils`

install build tools
`sudo apt-get install git wget cmake build-essential`

Prepare Linux Backend and the Dev. Environment (unplug any connected realsense cameras)
`sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at`

Install librealsense
`cd third-part/librealsense`
`git checkout v2.56.5`
`./scripts/setup_udev_rules.sh`
`mkdir build && cd build`
`cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=true -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=$(which python)`

Start build
`sudo make uninstall && make clean && make -j8 && sudo make install`

Test realsense-viewer
`realsense-viewer`

