# DX-RT NPU driver issue
## How to get/install kernel header for NPU driver installation:
- There are two options - local build or pre-built package
   ```bash
   # Option 1: local build
   sudo apt update
   sudo apt install git build-essential bison flex libssl-dev dwarves

   git clone https://github.com/orangepi-xunlong/orangepi-build.git
   cd orangepi-build
   ./build.sh >> Kernel package >> Do not change the kernel configuration >> orangepi5plus >> Current Recommended
   cd output/debs
   sudo dpkg -i linux-headers-current-rockchip-rk3588_1.2.0_arm64.deb

   # Option 2: use a prebuilt package
   wget https://github.com/user-attachments/files/22798846/linux-headers-6.1.x.tar.gz
   tar xzvf linux-headers-6.1.x.tar.gz
   sudo dpkg -i linux-headers-current-rockchip-rk3588_1.2.0_arm64.deb
   ```

# DX-STREAM issues
## How to fix GStreamer version conflicts:
1. Check and unhold all GStreamer packages:
   ```bash
   apt-mark showhold
   sudo apt-mark unhold \
     gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
     gstreamer1.0-plugins-bad gstreamer1.0-plugins-bad-apps gstreamer1.0-plugins-ugly \
     gstreamer1.0-libav gstreamer1.0-opencv gstreamer1.0-wpe \
     libgstreamer1.0-0 libgstreamer1.0-dev gir1.2-gstreamer-1.0 \
     libgstreamer-plugins-base1.0-0 libgstreamer-plugins-base1.0-dev \
     libgstreamer-plugins-good1.0-0 libgstreamer-plugins-good1.0-dev \
     libgstreamer-plugins-bad1.0-0 libgstreamer-plugins-bad1.0-dev \
     libgstreamer-opencv1.0-0 gir1.2-gst-plugins-bad-1.0 2>/dev/null || true
   ```
2. Temporarily remove the packages that enforce strict version locks:
   ```bash
   sudo apt purge -y \
     libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
     libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev \
     gstreamer1.0-opencv libgstreamer-opencv1.0-0 gstreamer1.0-wpe gir1.2-gst-plugins-bad-1.0
   sudo apt autoremove -y
   ```
3. Fix broken dependencies and update all base GStreamer libraries:
   ```bash
   sudo apt update
   sudo apt -t jammy-updates install -y --allow-downgrades --allow-change-held-packages \
     gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly gstreamer1.0-libav \
     gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-bad-apps \
     libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
     libgstreamer-plugins-good1.0-0 libgstreamer-plugins-bad1.0-0
   ```
4. Reinstall dev headers and optional plugin packages:
   ```bash
   sudo apt -t jammy-updates install -y --allow-downgrades \
     libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
     libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev \
     gstreamer1.0-opencv libgstreamer-opencv1.0-0 gstreamer1.0-wpe gir1.2-gst-plugins-bad-1.0
   ```
