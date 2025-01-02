#include <Adafruit_NeoPixel.h>

#define LED_PIN 6 // Neopixel control pin
#define LED_COUNT 8 // Number of NeoPixels

// Initialize Neopixel strip
Adafruit_NeoPixel strip(LED_COUNT, LED_PIN);

// Define NeoPixel white light color
uint32_t white = strip.Color(180, 180, 180);

void setup() {
  // put your setup code here, to run once:
  strip.begin();
  strip.show(); // Initialize all pixels to 'off'
}

void loop() {
  // put your main code here, to run repeatedly:
  strip.fill(white); // Turn all pixels white
  strip.show(); // Push data to NeoPixel
}
