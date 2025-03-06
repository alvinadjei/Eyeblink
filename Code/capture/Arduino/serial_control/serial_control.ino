#include <Adafruit_NeoPixel.h>

#define BUZZER_PIN 3 // Buzzer control pin
#define PUFF_PIN 5 // Airpuff control pin
#define PIX_PIN 6 // Neopixel control pin
#define PIX_COUNT 8 // Number of NeoPixels
#define IR_LED_1 9 // Side IR LED control pin
#define IR_LED_2 10 // Top IR LED control pin
#define IR_LED_3 11 // Head-on IR LED control pin

// Define constants
int csDuration = 280; // 280 ms duration for CS
int ISI = 250; // 250 ms b/w CS onset and US onset
int usDuration = 25; // 25 ms duration for US

// Initialize Neopixel strip
Adafruit_NeoPixel strip(PIX_COUNT, PIX_PIN);

// Define NeoPixel white light color
uint32_t white = strip.Color(255, 255, 255);

// Define starting LED brightnesses
int brightness_1 = 255; // Side IR LED
int brightness_2 = 255; // Top IR LED
int brightness_3 = 255; // Head-on IR LED

// Keep track of whether neopixels are on or off
bool pixOn = false;

void setup() {
  // Setup neopixel strip
  strip.begin();
  strip.show(); // Initialize all pixels to 'off'
  
  // Initialize control pins
  pinMode(BUZZER_PIN, OUTPUT); // Buzzer
  pinMode(PUFF_PIN, OUTPUT); // Air puff

  // Initialize serial communication
  Serial.begin(9600);
}

void loop() {

  // Write brightness values to IR LED control pins
  analogWrite(IR_LED_1, brightness_1); // side
  analogWrite(IR_LED_2, brightness_2); // top
  analogWrite(IR_LED_3, brightness_3); // head-on
  
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

    // Command to turn down head-on IR brightness
    if (command == 'j') { 
      if (brightness_3 > 0) {
        brightness_3 -= 5;
        Serial.println(brightness_3);
      } else {
        Serial.println(-1);
      }
    }

    // Command to turn up head-on IR brightness
    if (command == 'k') { 
      if (brightness_3 < 255) {
        brightness_3 += 5;
        Serial.println(brightness_3);
      } else {
        Serial.println(-1);
      }
    }

    // Command to run trial
    if (command == 'T') {
      // Conditioned stimulus
      tone(BUZZER_PIN, 10000);
      Serial.println("d");
      delay(ISI); // Wait 250 ms before starting US

      // Unconditioned stimulus
      digitalWrite(PUFF_PIN, HIGH); // Start airpuff
      delay(usDuration); // Duration of the air puff (25 ms)
      digitalWrite(PUFF_PIN, LOW);
      delay(csDuration - ISI - usDuration); // Finish playing tone
      noTone(BUZZER_PIN);
    }
  }
}
