#include <Adafruit_NeoPixel.h>

#define BUZZER_PIN 3   // Buzzer control pin
#define PUFF_PIN 5     // Airpuff control pin
#define PIX_PIN 6      // Neopixel control pin
#define PIX_COUNT 8    // Number of NeoPixels
#define IR_LED_1 9     // Side IR LED control pin
#define IR_LED_2 10    // Top IR LED control pin
#define IR_LED_3 11    // Head-on IR LED control pin
#define POT_PIN A0     // Potentiometer pin
#define BUTTON_PIN 12  // Button pin

// Define constants
int csDuration = 350;  // 350 ms duration for CS
int ISI = 300;         // 300 ms b/w CS onset and US onset
int usDuration = 50;   // 50 ms duration for US

// Define starting LED brightnesses
int brightness_1 = 255;  // Side IR LED
int brightness_2 = 255;  // Top IR LED
int brightness_3 = 255;  // Head-on IR LED

// Default buzzer frequency
int freq;

// Keep track of whether neopixels are on or off
bool pixOn = true;

// Define NeoPixel white light color
int pixLevel = 255;

// Initialize Neopixel strip
Adafruit_NeoPixel strip(PIX_COUNT, PIX_PIN);
uint32_t white = strip.Color(pixLevel, pixLevel, pixLevel);

// Button debouncing to prevent multiple triggers per press
bool lastButtonState = HIGH;
bool currentButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

void setup() {
  // Setup neopixel strip
  strip.begin();
  strip.show();  // Initialize all pixels to 'off'

  // Initialize control pins
  pinMode(BUZZER_PIN, OUTPUT);        // Buzzer
  pinMode(PUFF_PIN, OUTPUT);          // Air puff
  pinMode(IR_LED_1, OUTPUT);          // side LED
  pinMode(IR_LED_2, OUTPUT);          // top LED
  pinMode(IR_LED_3, OUTPUT);          // head-on LED(s)
  pinMode(BUTTON_PIN, INPUT_PULLUP);  // Neopixels brightness button

  // Initialize serial communication
  Serial.begin(9600);
}

void loop() {

  // Write brightness values to IR LED control pins
  analogWrite(IR_LED_1, brightness_1);  // side
  analogWrite(IR_LED_2, brightness_2);  // top
  analogWrite(IR_LED_3, brightness_3);  // head-on

  // Read potentiometer and set brightnesses
  int potVal = analogRead(POT_PIN);
  if (potVal < 341) {
    brightness_1 = 0;
    brightness_2 = 255;
  } else if (potVal > 682) {
    brightness_1 = 255;
    brightness_2 = 0;
  } else {
    brightness_1 = 255;
    brightness_2 = 255;
  }

  // Read button state and change neopixels brightness accordingly
  bool reading = digitalRead(BUTTON_PIN);
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != currentButtonState) {
      currentButtonState = reading;
      // Button pressed (HIGH --> LOW)
      if (currentButtonState == LOW) {
        if (pixLevel < 255) {
          pixLevel += 51;
        } else
        pixLevel = 0;
      }
    }
  }
  lastButtonState = reading;

  // Set neopixel brightness on each loop
  uint8_t correctedLevel = strip.gamma8(pixLevel);  // evenly spaces the perceived brightness, not just the PWM signal sent to neopixels
  uint32_t white = strip.Color(correctedLevel, correctedLevel, correctedLevel);

  // Turn overhead light on/off depending on value of pixOn
  if (pixOn) {          // switch pixels on
    strip.fill(white);  // Turn all pixels white
    strip.show();       // Push data to NeoPixel

  } else {  // switch pixels off
    // Turn off all pixels
    strip.clear();
    strip.show();
  }

  if (Serial.available() > 0) {
    char command = Serial.read();

    // Command to toggle houselight
    if (command == 'h') {
      pixOn = !pixOn;  // toggle pixOn bool
      if (pixOn) {
        Serial.println("on");
      } else {
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
    if (command == 'T' || command == 'F') {
      // Conditioned stimulus
      if (command == 'F') {
        // Wait for the rest of the int to arrive
        freq = Serial.parseInt(SKIP_NONE, 'F') * 1000;  // Reads the number after 'F'
      } else {
        freq = 10000;
      }

      // Begin CS-US
      Serial.println("d");
      tone(BUZZER_PIN, freq);

      delay(ISI);  // Wait for ISI before starting US

      // If command is 'F', don't do airpuff
      if (command == 'T') {
        // Unconditioned stimulus
        digitalWrite(PUFF_PIN, HIGH);  // Start airpuff
        delay(usDuration);             // Duration of the air puff
        digitalWrite(PUFF_PIN, LOW);
      } else {              // no airpuff
        delay(usDuration);  // Duration of the air puff
      }

      delay(csDuration - ISI - usDuration);  // Finish playing tone
      noTone(BUZZER_PIN);
    }
  }
}
