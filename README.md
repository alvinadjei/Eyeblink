# Eyeblink Experiment

This repository contains code and instructions for setting up an eyeblink conditioning experiment in mice. In this case, the conditioned stimulus is a short musical tone, 
and the unconditioned stimulus is a short puff of air to the mouse's eye.

This page is a work in progress, and will be updated as the project progresses.

Author(s): Alvin Adjei, University of California San Francisco

## Requirements
### Software
The code for this project is written in Python and in the Arduino IDE. The required Python libraries and their versions are specified in _requirements.txt_.
To install them, run <code>python -m pip install -r requirements.txt</code>
in the terminal in the project's root directory.

The arduino code requires the Adafruit NeoPixel library, which can be installed via the Library Manager in the Arduino IDE.

### Hardware
- Designed to work on modern Windows computers, untested on Mac
- 1 $\times$
  <a href="https://store.arduino.cc/products/arduino-uno-rev3?srsltid=AfmBOophdIvm8RfX5799wr4zovlr1sxV1jH-H7QClWuCym0v6gZN2AgC">
    Arudino Uno
  </a>
  or similar microcontroller (we used an
  <a href="https://www.adafruit.com/product/2488">
    Adafruit Metro Board
  </a>)
- 1 $\times$ Picospritzer III or similar device that can accept Arduino HIGH as input signal and outputs an air puff
- 1 $\times$
  <a href="https://www.adafruit.com/product/1426">
    neopixel strip
  </a>
- 5 $\times$ infrared LEDs (the ones we used emit light w/ 850nm wavelength)
- 5 $\times$ 220 $\Omega$ resistors
- 1 $\times$ passive piezoeelectric buzzer
- 1 $\times$
  <a href="https://www.edmundoptics.com/p/allied-vision-alvium-1800-u-120m-13-12mp-c-mount-usb-31-monochrome-camera/48262/">
    Allied Vision Alvium 1800 U-120m Monochrome Camera</a>,
  or similar camera with ___no___ infrared filter and compatible lens (in this case, for a 1/3" sensor)
  - Camera drivers and software are <a href="https://www.alliedvision.com/en/products/software/" target="_blank">here</a>
  - Python API manual is <a href="https://docs.alliedvision.com/Vimba_X/Vimba_X_DeveloperGuide/pythonAPIManual.html">here</a>
- Infrared mirror
- Other materials you see in the image below come from Thorlabs:

### Images of setup here

