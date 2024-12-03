int puffPin = 12; // Replace with the pin you're using

void setup() {
  pinMode(puffPin, OUTPUT);
}

void loop() {
  digitalWrite(puffPin, HIGH);
  delay(500);  // Duration of the air puff (250 ms)
  digitalWrite(puffPin, LOW);
  delay(500);  // Rest between air puffs (250 ms)
}
