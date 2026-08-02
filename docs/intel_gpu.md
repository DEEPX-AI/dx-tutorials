# Intel GPU Support

## How to install Intel GPU driver - Ubuntu

### For Ubuntu Desktop 26.04, 25.10, or 24.04:
Intel offers the intel-graphics Personal Package Archive (PPA). This PPA provides early access to the latest packages, along with additional tools and features, such as EU debugging. Follow these steps to install the intel-graphics PPA and the required compute and media packages.

1. Refresh the local package index and install the package for managing software repositories.
```sh
sudo apt-get update
sudo apt-get install -y software-properties-common
```

2. Add the `intel-graphics` PPA.
```sh
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
```

3. Install the compute-releated packages.
```sh
sudo apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc
```

4. Install the media-related packages.
```sh
sudo apt-get install -y intel-media-va-driver-non-free libmfx-gen1.2 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo gstreamer1.0-vaapi
```

5. Install the Intel GPU tool.
```sh
sudo apt-get install intel-gpu-tools
```

6. Verifying installation
To verify that the kernel and compute drivers are installed and functional, run clinfo:
```sh
vainfo
clinfo | grep "Device Name"
sudo intel_gpu_top
```

You should see the Intel graphics product device names listed. If they do not appear, ensure you have permissions to access /dev/dri/renderD*. This typically requires your user to be in the render group:
```sh
sudo gpasswd -a ${USER} render
newgrp render
```

Refer to [Intel guidance (Installing Packages from Ubuntu PPA)](https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html)
