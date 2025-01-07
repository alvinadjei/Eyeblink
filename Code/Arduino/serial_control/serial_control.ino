#include <Adafruit_NeoPixel.h>

#define PIX_PIN 6 // Neopixel control pin
#define PIX_COUNT 8 // Number of NeoPixels
#define IR_LED_1 9 // Side IR LED Control Pin
#define IR_LED_2 10 // Top IR LED Control Pin
#define PUFF_PIN 11 // Airpuff control pin

// Initialize Neopixel strip
Adafruit_NeoPixel strip(PIX_COUNT, PIX_PIN);

// Define NeoPixel white light color
uint32_t white = strip.Color(180, 180, 180);

// Define starting LED brightnesses
int brightness_1 = 130; // Side IR LED
int brightness_2 = 130; // Top IR LED

// Keep track of whether neopixels are on or off
bool pixOn = false;

void setup() {
  // Setup neopixel strip
  strip.begin();
  strip.show(); // Initialize all pixels to 'off'
  
  // Initialize airpuff control pin
  pinMode(PUFF_PIN, OUTPUT);

  // Initialize serial communication
  Serial.begin(9600);
}

void loop() {

  // Write brightness values to IR LED control pins
  analogWrite(IR_LED_1, brightness_1); // side
  analogWrite(IR_LED_2, brightness_2); // top

  if (Serial.available() > 0) {
    char command = Serial.read();
    
    // Command to toggle houselight
    if (command == 'h') { 
      pixOn = !pixOn; // toggle pixOn bool

      if (pixOn) { // switch pixels on
        strip.fill(white); // Turn all pixels white
        strip.show(); // Push data to NeoPixel
        Serial.println("on");
      } else { // switch pixels off
        // Turn off all pixels
        strip.clear();
        strip.show();
        Serial.println("off");
      }
    }

    // Command to turn down side IR brightness
    if (command == 'a') { 
      if (brightness_1 > 0) {
        brightness_1 -= 5;
        Serial.println(brightness_1);
      } else {
        Serial.println(-1);
      }
    }

    // Command to turn up side IR brightness
    if (command == 'd') { 
      if (brightness_1 < 255) {
        brightness_1 += 5;
        Serial.println(brightness_1);
      } else {
        Serial.println(-1);
      }
    }

    // Command to turn down top IR brightness
    if (command == 's') { 
      if (brightness_2 > 0) {
        brightness_2 -= 5;
        Serial.println(brightness_2);
      } else {
        Serial.println(-1);
      }
    }

    // Command to turn up top IR brightness
    if (command == 'w') { 
      if (brightness_2 < 255) {
        brightness_2 += 5;
        Serial.println(brightness_2);
      } else {
        Serial.println(-1);
      }
    }

    // Command to trigger a puff
    if (command == 'p') { 
      digitalWrite(PUFF_PIN, HIGH);
      delay(25);  // Duration of the air puff (25 ms)
      digitalWrite(PUFF_PIN, LOW);
      Serial.println("d");
    }
  }
}
