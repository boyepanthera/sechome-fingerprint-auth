# SECHOME

Fingerprint Recognition System with R307 and DeepPrint

This system combines hardware fingerprint sensing using the R307 sensor with deep learning-based verification using DeepPrint.

## Hardware Requirements

1. R307 Fingerprint Sensor:

   - Operating voltage: 3.3V to 5V
   - Purchase link: [Amazon - R307 Fingerprint Sensor](https://www.amazon.com/HiLetgo-Optical-Fingerprint-Reader-Arduino/dp/B07G8RJBB3)

2. USB to TTL Converter:

   - FT232RL or similar
   - Purchase link: [Amazon - USB TTL Converter](https://www.amazon.com/HiLetgo-FT232RL-Converter-Adapter-Arduino/dp/B00IJXZQ7C)

3. Additional Items:
   - Jumper wires (female-to-female)
   - Mini USB cable
   - Computer/Laptop with USB port

## Wiring Diagram

```

R307 Sensor USB-TTL Converter

---

VCC (Red) --> 3.3V/5V
GND (Black) --> GND
TX (Green) --> RX
RX (White) --> TX

                  USB-TTL
                  -------
                    USB  -->  Computer

```

## System Requirements

- Python 3.8 or higher
- Windows/Linux/MacOS
- 4GB RAM minimum
- USB port
- Internet connection (for initial package installation)

## Software Installation

1. Install Python Dependencies:

```bash
pip install pyserial
pip install torch
pip install numpy
pip install Pillow
pip install simpleaudio
```

2. Sound Files Setup:

   - Create two WAV files:
     - success.wav (for successful recognition)
     - failure.wav (for failed recognition)
   - Place them in the project directory

3. Port Configuration:
   - Windows: Update port to 'COM3' (or similar)
   - Linux: Use '/dev/ttyUSB0'
   - Mac: Use '/dev/tty.usbserial-\*'

## Initial Setup

1. Hardware Connection:

   - Connect R307 to USB-TTL following wiring diagram
   - Connect USB-TTL to computer
   - Note assigned COM port

2. Port Permissions (Linux/Mac):

```bash
sudo chmod 666 /dev/ttyUSB0  # Replace with your port
```

3. Test Connection:

```bash
python main.py
```

## Usage Instructions

1. Starting the System:
   - Run the program:

```bash
python main.py
```

2. Available Options:

   - 1: Enroll new fingerprint
   - 2: Verify fingerprint
   - 3: Delete fingerprint
   - 4: Empty database
   - 5: Save features database
   - 6: Exit

3. Enrollment Process:

   - Select option 1
   - Enter unique ID (1-999)
   - Follow on-screen instructions
   - Place finger twice when prompted

4. Verification Process:
   - Select option 2
   - Place finger on sensor
   - Wait for result and sound feedback

## Troubleshooting Guide

1. Connection Issues:

   - Check USB connections
   - Verify power supply
   - Confirm port settings
   - Test different USB ports

2. Recognition Problems:

   - Clean sensor surface
   - Clean finger surface
   - Check finger placement
   - Re-enroll if necessary

3. Software Issues:
   - Check Python version
   - Verify all packages installed
   - Check port permissions
   - Review log files

## Safety and Maintenance

1. Hardware Care:

   - Keep sensor clean
   - Avoid static discharge
   - Check connections regularly
   - Don't exceed voltage ratings

2. Data Security:

   - Regular database backups
   - Secure storage of fingerprint data
   - Monitor access logs

3. Regular Maintenance:
   - Weekly sensor cleaning
   - Monthly connection check
   - Regular software updates
   - Log file review

## System Architecture

1. Hardware Layer:

   - R307 Sensor: Captures fingerprint
   - USB-TTL: Communication interface
   - Sound system: Feedback mechanism

2. Software Layer:
   - Python interface
   - DeepPrint processing
   - Database management
   - Sound feedback system

## File Structure

```
project/
│
├── main.py    # Main program
├── success.wav             # Success sound
├── failure.wav            # Failure sound
├── fingerprint_features.npy # Feature database
└── logs/                  # System logs
```

## Support and Updates

1. Issue Reporting:

   - Check troubleshooting first
   - Review logs
   - Document reproduction steps

2. System Updates:
   - Regular package updates
   - Feature improvements
   - Security patches

## License

This project is released under the MIT License. See LICENSE file for details.

## Disclaimer

This system is for educational and development purposes. Implement appropriate security measures for production use.
