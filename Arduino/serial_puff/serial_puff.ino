int puffPin = 11; // Replace with the pin you're using

void setup() {
  pinMode(puffPin, OUTPUT);
  Serial.begin(9600); // Initialize serial communication
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'p') { // Command to trigger a puff
      digitalWrite(puffPin, HIGH);
      delay(25);  // Duration of the air puff (25 ms)
      digitalWrite(puffPin, LOW);
      Serial.println("d");
    }
  }
}
