# AquaSense AI - Physical Edge Sensor Wiring & Hardware Blueprint

This document details the hardware schematics and electrical connections for deploying physical edge sensors with an ESP32 microcontroller streaming to the AquaSense AI platform.

---

## 1. Hardware Bill of Materials (BOM)

| Component | Part / Model Number | Interface | Operating Voltage |
| :--- | :--- | :--- | :--- |
| **Microcontroller** | ESP32 DevKit v1 (30-pin) | Micro-USB / 2.4GHz Wi-Fi | 3.3V / 5.0V |
| **pH Sensor** | DFRobot Analog pH Meter Pro (SEN0161) | Analog (0 - 3.3V) | 5.0V (VCC) / 3.3V Signal |
| **Turbidity Sensor** | DFRobot Gravity Turbidity Sensor (SEN0189) | Analog (0 - 4.5V via Divider) | 5.0V |
| **Temperature Sensor**| Dallas DS18B20 Waterproof Probe | Digital (1-Wire) | 3.3V - 5.0V |
| **Resistor** | 4.7 kΩ (Pull-up for DS18B20) | Axial 1/4W | N/A |
| **Power Supply** | 5V 2A Regulated DC Adapter or LiPo | Barrel Jack / USB | 5.0V DC |

---

## 2. Pinout Connection Matrix

### A. DFRobot Analog pH Sensor
* **VCC:** Connect to `VIN` (5V external rail).
* **GND:** Connect to ESP32 `GND` (common ground).
* **Signal (Analog):** Connect to `GPIO 34` (ADC1_CH6).  
  *(Note: ADC1 pins on the ESP32 remain functional when Wi-Fi is active; do not use ADC2 pins such as GPIO 2, 4, 12-15 for analog readings).*

### B. Optical Turbidity Sensor
* **VCC:** Connect to `VIN` (5V).
* **GND:** Connect to `GND`.
* **Signal (Analog):** Connect to `GPIO 35` (ADC1_CH7) via a 2:3 voltage divider ($10\text{ k}\Omega$ and $20\text{ k}\Omega$) if signal exceeds $3.3\text{ V}$.

### C. DS18B20 Digital Temperature Probe
* **VDD (Red):** Connect to `3V3`.
* **GND (Black):** Connect to `GND`.
* **Data (Yellow/Blue):** Connect to `GPIO 4`.
* **Pull-up Resistor:** Place a $4.7\text{ k}\Omega$ resistor between `GPIO 4` (Data) and `3V3`.

---

## 3. Calibration Instructions

1. **pH Two-Point Buffer Calibration:**
   * Immerse probe in pH 7.00 buffer solution at 25°C. Adjust potentiometer on signal conditioning board until output reads $1.65\text{ V}$ (midpoint of ESP32 ADC).
   * Rinse with deionized water; immerse in pH 4.01 buffer to record slope $\Delta V / \Delta\text{pH}$.
2. **Turbidity Zero-Point Calibration:**
   * Place probe in clean deionized water ($0\text{ NTU}$). Record voltage $V_{\text{clean}} \approx 4.1\text{ V}$.
   * Verify that muddy water causes voltage to drop proportionally to light scattering.
