#include <Adafruit_NeoPixel.h>

#define BUZZER_PIN 3 // Buzzer control pin

#define PIX_PIN 6 // Neopixel control pin
#define PIX_COUNT 8 // Number of NeoPixels

// Initialize Neopixel strip
Adafruit_NeoPixel strip(PIX_COUNT, PIX_PIN);

// Define NeoPixel white light color
uint32_t white = strip.Color(255, 255, 255);

void setup() {
  // Setup neopixel strip
  strip.begin();
  strip.show(); // Initialize all pixels to 'off'
  
  // Initialize buzzer pin
  pinMode(BUZZER_PIN, OUTPUT); // Buzzer

  // Initialize serial communication
  Serial.begin(9600);
}

void loop() {

  strip.fill(white); // Turn all pixels white
  strip.show(); // Push data to NeoPixel
  Serial.println("on");
  
  // Play buzzer for 3 seconds
  tone(BUZZER_PIN, 10000);
  delay(3000);

  // Turn off buzzer for 1 second
  noTone(BUZZER_PIN);
  delay(1000);
}
