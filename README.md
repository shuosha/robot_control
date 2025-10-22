# For real robot env installation

set up python env:
`uv venv --python 3.11 isaac`
`source isaac/bin/activate`
`uv sync`

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
`cd src/third_party/librealsense`
`git checkout v2.56.5`
`./scripts/setup_udev_rules.sh`
`mkdir build && cd build`
`cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=true -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=$(which python)`
`cd ../../../`
Start build
`sudo make uninstall && make clean && make -j8 && sudo make install`
Test realsense-viewer
`realsense-viewer`

# Gello setup
Follow instructions on DYNAMIXEL Wizard 2.0 to download software
https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/

Change permission:

`sudo chmod 775 DynamixelWizard2Setup_x64`

Install program

`./DynamixelWizard2Setup_x64`

Click next to proceed installation

Add account to dialout group

`sudo usermod -aG dialout <your_account_id>`

reboot

`sudo reboot`


clone and setup gello

`python scripts/gello_get_offset.py --start-joints 0 -0.79 0 0.52 0 1.31 0 --joint-signs 1 1 1 1 1 1 1 --port /dev/ttyUSB0`

Change offset, check port, check baudrate, check robot ip