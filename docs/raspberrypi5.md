# Raspberry Pi 5 PCIe Gen3 x1 Configuration & Verification Guide

By default, the PCIe interface on the Raspberry Pi 5 is configured to operate at **Gen2 speeds (5 GT/s)**. To maximize the bandwidth for high-performance NVMe SSDs or other PCIe peripherals, you can force the interface to operate at **Gen3 speeds (8 GT/s)**. 

> **Note:** While the Raspberry Pi Foundation does not officially certify Gen3 speeds, it is widely known to work stably with high-quality FPC cables and compatible expansion boards.

## 1. Enabling PCIe Gen3

1. Open the terminal and edit the `config.txt` file:
   ```bash
   sudo nano /boot/firmware/config.txt
   ```

2. Add the following line to the very bottom of the file:
   ```Ini, TOML
   # Enable PCIe Gen3 x1
   dtparam=pciex1_gen=3
   ```

3. Save the file (Ctrl+O, Enter, Ctrl+X) and reboot the system:
   ```bash
   sudo reboot
   ```

## 2. Verifying the Configuration (lspci)

After rebooting, you should verify if the system has successfully negotiated a Gen3 link with your PCIe device.

1. Run the `lspci` command with the verbose flag to output detailed device information:
   ```bash
   sudo lspci -vvv
   ```
   > Note: If the command is not found, install the utility by running `sudo apt install pciutils`.

2. Locate the section for your connected PCIe device (e.g., Non-Volatile memory controller or PCIe Bridge). You need to check two specific parameters: `LnkCap` (Link Capabilities) and `LnkSta` (Link Status).

   Example of a successful Gen3 link output:
   ```Plaintext
   Capabilities: [c0] Express (v2) Endpoint, MSI 00
    ...
    LnkCap: Port #0, Speed 8GT/s, Width x1, ASPM L1, Exit Latency L1 <64us
    ...
    LnkSta: Speed 8GT/s (ok), Width x1 (ok)
   ```

## 3. Reference output for `sudo lspci -vvv` on Raspberry Pi 5 with DX-M1 module

Below is a reference output showing the DX-M1 module successfully recognized. The Link Capabilities (LnkCap) and actual Link Status (LnkSta) are highlighted.

```diff
+0001:01:00.0 Processing accelerators: DEEPX Co., Ltd. DX_M1 (rev 01)
	Subsystem: DEEPX Co., Ltd. DX_M1
	Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR+ FastB2B- DisINTx+
	Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
	Latency: 0
	Interrupt: pin A routed to IRQ 177
	Region 0: Memory at 1800000000 (64-bit, prefetchable) [size=8M]
	Region 2: Memory at 1b80000000 (32-bit, non-prefetchable) [size=1M]
	Region 3: Memory at 1b80100000 (32-bit, non-prefetchable) [size=64K]
	Region 4: Memory at 1b80120000 (32-bit, non-prefetchable) [size=4K]
	Region 5: Memory at 1b80110000 (32-bit, non-prefetchable) [size=64K]
	Capabilities: [40] Power Management version 3
		Flags: PMEClk- DSI- D1+ D2- AuxCurrent=375mA PME(D0+,D1+,D2-,D3hot+,D3cold-)
		Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
	Capabilities: [50] MSI: Enable+ Count=8/16 Maskable+ 64bit+
		Address: 000000fffffff000  Data: 0008
		Masking: 0000ffe0  Pending: 00000380
	Capabilities: [70] Express (v2) Endpoint, IntMsgNum 0
		DevCap:	MaxPayload 256 bytes, PhantFunc 0, Latency L0s unlimited, L1 unlimited
			ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset+ SlotPowerLimit 0W TEE-IO-
		DevCtl:	CorrErr+ NonFatalErr+ FatalErr+ UnsupReq+
			RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop- FLReset-
			MaxPayload 256 bytes, MaxReadReq 512 bytes
		DevSta:	CorrErr+ NonFatalErr- FatalErr- UnsupReq- AuxPwr- TransPend-
+		LnkCap:	Port #0, Speed 8GT/s, Width x4, ASPM L1, Exit Latency L1 <16us
			ClockPM- Surprise- LLActRep- BwNot- ASPMOptComp+
		LnkCtl:	ASPM L1 Enabled; RCB 64 bytes, LnkDisable- CommClk+
			ExtSynch- ClockPM- AutWidDis- BWInt- AutBWInt-
+		LnkSta:	Speed 8GT/s, Width x1 (downgraded)
			TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
		DevCap2: Completion Timeout: Not Supported, TimeoutDis+ NROPrPrP- LTR+
			 10BitTagComp- 10BitTagReq- OBFF Via message, ExtFmt+ EETLPPrefix+, MaxEETLPPrefixes 4
			 EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
			 FRS+ TPHComp+ ExtTPHComp-
			 AtomicOpsCap: 32bit- 64bit- 128bitCAS-
```

## 💡 PCIe Speed Comparison (x1 Lane)

Understanding the different PCIe generations helps illustrate the performance boost you get from this configuration. The Raspberry Pi 5 has a single PCIe lane (**x1**).

| PCIe Generation | Transfer Rate (GT/s) | Max Theoretical Bandwidth (MB/s) | Encoding Overhead |
| :--- | :--- | :--- | :--- |
| **Gen1** | 2.5 GT/s | ~250 MB/s | 8b/10b (20% overhead) |
| **Gen2** (RPi 5 Default) | 5.0 GT/s | ~500 MB/s | 8b/10b (20% overhead) |
| **Gen3** (Upgraded) | 8.0 GT/s | **~985 MB/s** | 128b/130b (~1.5% overhead) |

*By upgrading from Gen2 to Gen3, you effectively double the available bandwidth, allowing devices like the DX-M1 module or fast NVMe drives to operate much closer to their maximum potential.*


## Dual HAT case (made by Seeed Studio)
Need to set additional configuraitons.
   ```bash
   dtparam=pciex1
   dtparam=pciex1_gen=3
   dtoverlay=pciex1-compat-pi5,no-mip,mmio-hi 
   ```
For more details, refer to [this wiki page](https://wiki.seeedstudio.com/raspberry_pi_5_uses_pcie_hat_dual_hat/).
