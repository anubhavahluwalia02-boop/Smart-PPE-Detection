/*
 * =========================================================================================
 *  SMART PPE COMPLIANCE DETECTION SYSTEM - ESP32 FIRMWARE
 * =========================================================================================
 *  Board: ESP32 Dev Module (or ESP32-WROOM-32)
 *  Baud Rate: 115200 bps
 *  Hardware Connections:
 *    - Green LED    -> GPIO 18 (with 220Ω current-limiting resistor to GND)
 *    - Red LED      -> GPIO 19 (with 220Ω current-limiting resistor to GND)
 *    - Active Buzzer-> GPIO 21 (Positive pin to GPIO 21, Negative to GND)
 *
 *  Supported Serial Commands (ASCII terminated by \n or \r):
 *    1. "SAFE"             -> Compliant state: Green LED ON, Red LED OFF, Buzzer OFF
 *    2. "VIOLATION"        -> Non-compliant state: Green LED OFF, Red LED ON, Buzzer ALARM
 *    3. "PING"             -> Heartbeat query: Responds "PONG"
 *    4. "IDLE" / "CLEAR"   -> Standby: All outputs OFF
 *    5. "P18:1,P19:0,P21:0"-> Direct pin override command
 *    6. "STATUS"           -> Prints current pin states and system uptime
 * =========================================================================================
 */

// Hardware Pin Definitions
const int PIN_GREEN_LED = 18;  // Safe / Compliant indicator
const int PIN_RED_LED   = 19;  // Non-compliant / Breach indicator
const int PIN_BUZZER    = 21;  // Audible piezoelectric alarm

// State tracking
bool greenState = false;
bool redState = false;
bool buzzerState = false;
bool isViolationActive = false;

// Non-blocking buzzer pulsing parameters during active violation
unsigned long lastBuzzerToggleTime = 0;
const unsigned long BUZZER_INTERVAL_MS = 250; // 250ms ON, 250ms OFF pulse
bool buzzerPulseHigh = false;

// Serial incoming command buffer
String inputBuffer = "";

void setOutputs(bool green, bool red, bool buzzer) {
  greenState = green;
  redState = red;
  buzzerState = buzzer;

  digitalWrite(PIN_GREEN_LED, green ? HIGH : LOW);
  digitalWrite(PIN_RED_LED, red ? HIGH : LOW);
  digitalWrite(PIN_BUZZER, buzzer ? HIGH : LOW);
}

void parsePinCommand(String cmd) {
  // Parses "P18:1,P19:0,P21:1"
  int p18Idx = cmd.indexOf("P18:");
  int p19Idx = cmd.indexOf("P19:");
  int p21Idx = cmd.indexOf("P21:");

  if (p18Idx != -1) {
    char val = cmd.charAt(p18Idx + 4);
    greenState = (val == '1');
    digitalWrite(PIN_GREEN_LED, greenState ? HIGH : LOW);
  }
  if (p19Idx != -1) {
    char val = cmd.charAt(p19Idx + 4);
    redState = (val == '1');
    digitalWrite(PIN_RED_LED, redState ? HIGH : LOW);
  }
  if (p21Idx != -1) {
    char val = cmd.charAt(p21Idx + 4);
    buzzerState = (val == '1');
    digitalWrite(PIN_BUZZER, buzzerState ? HIGH : LOW);
  }

  isViolationActive = (redState && buzzerState);
  Serial.printf("[ESP32] Pin Override Applied -> G(18):%d, R(19):%d, BZ(21):%d\n", greenState, redState, buzzerState);
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  // 1. SAFE command
  if (cmd.equalsIgnoreCase("SAFE") || cmd.indexOf("COMPLIANT") != -1) {
    isViolationActive = false;
    setOutputs(true, false, false);
    Serial.println("[ESP32] STATUS: SAFE (Green LED ON, Red OFF, Buzzer OFF)");
  }
  // 2. VIOLATION command
  else if (cmd.equalsIgnoreCase("VIOLATION") || cmd.indexOf("ALERT") != -1 || cmd.indexOf("CRITICAL") != -1) {
    isViolationActive = true;
    buzzerPulseHigh = true;
    lastBuzzerToggleTime = millis();
    digitalWrite(PIN_GREEN_LED, LOW);
    digitalWrite(PIN_RED_LED, HIGH);
    digitalWrite(PIN_BUZZER, HIGH);
    greenState = false;
    redState = true;
    buzzerState = true;
    Serial.println("[ESP32] STATUS: VIOLATION DETECTED (Red LED ON, Buzzer ALARM)");
  }
  // 3. Heartbeat PING query
  else if (cmd.equalsIgnoreCase("PING") || cmd.indexOf("CMD:PING") != -1) {
    Serial.println("PONG");
  }
  // 4. Standby / IDLE / CLEAR
  else if (cmd.equalsIgnoreCase("IDLE") || cmd.equalsIgnoreCase("CLEAR")) {
    isViolationActive = false;
    setOutputs(false, false, false);
    Serial.println("[ESP32] STATUS: IDLE (All Indicators OFF)");
  }
  // 5. Explicit Pin format: P18:1,P19:0...
  else if (cmd.startsWith("P18:") || cmd.indexOf("P18:") != -1) {
    parsePinCommand(cmd);
  }
  // 6. Status report
  else if (cmd.equalsIgnoreCase("STATUS")) {
    Serial.printf("[ESP32 STATUS] Uptime: %lu ms | G18: %s | R19: %s | BZ21: %s\n",
                  millis(),
                  greenState ? "ON" : "OFF",
                  redState ? "ON" : "OFF",
                  buzzerState ? "ON" : "OFF");
  }
  else {
    Serial.printf("[ESP32] Unknown Command Received: '%s'\n", cmd.c_str());
  }
}

void setup() {
  Serial.begin(115200);
  
  // Configure GPIO directions
  pinMode(PIN_GREEN_LED, OUTPUT);
  pinMode(PIN_RED_LED, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);

  // Initial state: all OFF
  setOutputs(false, false, false);

  // Power-On Self-Test (POST) diagnostic sequence
  delay(100);
  Serial.println("\n========================================================");
  Serial.println(" SMART PPE COMPLIANCE DETECTION - ESP32 CONTROLLER v1.0");
  Serial.println(" Pins Configured: Green=18, Red=19, Buzzer=21");
  Serial.println(" Running Power-On Self-Test (POST)...");

  // Blink Green
  digitalWrite(PIN_GREEN_LED, HIGH);
  delay(180);
  digitalWrite(PIN_GREEN_LED, LOW);

  // Blink Red
  digitalWrite(PIN_RED_LED, HIGH);
  delay(180);
  digitalWrite(PIN_RED_LED, LOW);

  // Short Buzzer Chirp
  digitalWrite(PIN_BUZZER, HIGH);
  delay(90);
  digitalWrite(PIN_BUZZER, LOW);

  Serial.println(" Self-Test Passed! System Ready.");
  Serial.println(" Listening for 'SAFE', 'VIOLATION', 'PING' on Serial (115200 baud)...");
  Serial.println("========================================================\n");
}

void loop() {
  // 1. Process incoming serial bytes
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        handleCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      if (inputBuffer.length() < 128) {
        inputBuffer += c;
      }
    }
  }

  // 2. Non-blocking buzzer pulsing during active violation
  if (isViolationActive) {
    unsigned long currentMillis = millis();
    if (currentMillis - lastBuzzerToggleTime >= BUZZER_INTERVAL_MS) {
      lastBuzzerToggleTime = currentMillis;
      buzzerPulseHigh = !buzzerPulseHigh;
      digitalWrite(PIN_BUZZER, buzzerPulseHigh ? HIGH : LOW);
    }
  }
}
