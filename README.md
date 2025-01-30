# Eyeblink Experiment

## Overview
This repository contains code and instructions for setting up an eyeblink conditioning experiment in mice. In this case, the conditioned stimulus is a short musical tone, 
and the unconditioned stimulus is a short puff of air to the mouse's eye.

This page is a work in progress, and will be updated as the project progresses.

Author(s): Alvin Adjei, University of California San Francisco

## Requirements
### Software
The code for this project is written in Python and in the Arduino IDE. The required Python libraries and their versions are specified in _requirements.txt_.
The arduino code requires the Adafruit NeoPixel library, which can be installed via the Library Manager in the Arduino IDE.

### Hardware
- Designed to work on modern Windows computers, untested on Mac
- 1 $\times$
  <a href="https://www.adafruit.com/product/1426" target="_blank">
    neopixel strip
  </a>
- 5 $\times$ infrared LEDs (the ones we used emit light w/ 850nm wavelength)
- 5 $\times$ 220 $\Omega$ resistors
- 1 $\times$ passive piezoeelectric buzzer
- 1 $\times$
  <a href="https://www.edmundoptics.com/p/allied-vision-alvium-1800-u-120m-13-12mp-c-mount-usb-31-monochrome-camera/48262/" target="_blank">
    Allied Vision Alvium 1800 U-120m, 1/3” 1.2MP C-Mount, USB 3.1 Monochrome Camera
  </a>
  - Camera drivers and software are <a href="https://www.alliedvision.com/en/products/software/" target="_blank">here</a>
  - Python API manual is <a href="https://docs.alliedvision.com/Vimba_X/Vimba_X_DeveloperGuide/pythonAPIManual.html" target="_blank">here</a>
- 
