// ======== MULTI-SENSOR SIGNAL READER ========
// Sensors: ECG, PPG, MEMS (Accelerometer), Bluetooth (HC-05 optional)
// Sends comma-separated values over Serial for 40s sampling

const int ecgPin = A0;     // ECG analog pin
const int ppgPin = A1;     // PPG analog pin
const int memsX = A2;      // MEMS X-axis
const int memsY = A3;      // MEMS Y-axis
const int memsZ = A4;      // MEMS Z-axis

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("ECG,PPG,MEMS_X,MEMS_Y,MEMS_Z");
}

void loop() {
  int ecg = analogRead(ecgPin);
  int ppg = analogRead(ppgPin);
  int x = analogRead(memsX);
  int y = analogRead(memsY);
  int z = analogRead(memsZ);

  Serial.print(ecg); Serial.print(",");
  Serial.print(ppg); Serial.print(",");
  Serial.print(x); Serial.print(",");
  Serial.print(y); Serial.print(",");
  Serial.println(z);

  delay(25);  // ~40Hz sampling (~1600 samples in 40s)
}
