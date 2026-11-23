/*
 * AquaSense AI - ESP32 Hardware Edge Multi-Sensor Telemetry Client
 * Target Microcontroller: ESP32 DevKit v1 / NodeMCU-32S
 * Sensors: Analog pH (DFRobot), Analog Optical Turbidity, OneWire DS18B20 Temp
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// Wi-Fi Credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// AquaSense AI API Endpoint
const char* serverUrl = "http://192.168.1.100:8000/api/v1/predict";

// Pin Definitions
#define PH_PIN 34           // ADC1_CH6 Analog Input (pH sensor)
#define TURBIDITY_PIN 35    // ADC1_CH7 Analog Input (Turbidity sensor)
#define ONE_WIRE_BUS 4      // Digital GPIO4 for DS18B20 temperature probe

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensor(&oneWire);

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  // Initialize sensors
  analogReadResolution(12); // 12-bit ADC (0 - 4095)
  tempSensor.begin();

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWi-Fi Connected successfully.");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

float readPH() {
  int rawADC = analogRead(PH_PIN);
  float voltage = rawADC * (3.3 / 4095.0);
  // Calibration equation: pH = 7.0 + ((2.5 - voltage) / 0.18)
  float phValue = 7.0 + ((1.65 - voltage) * 3.5);
  return constrain(phValue, 0.0, 14.0);
}

float readTurbidity() {
  int rawADC = analogRead(TURBIDITY_PIN);
  float voltage = rawADC * (3.3 / 4095.0);
  // Characteristic polynomial for turbidity in NTU:
  float ntu = -1120.4 * (voltage * voltage) + 5742.3 * voltage - 4352.9;
  if (ntu < 0) ntu = 0.0;
  return constrain(ntu, 0.0, 500.0);
}

float readTemperature() {
  tempSensor.requestTemperatures();
  float tempC = tempSensor.getTempCByIndex(0);
  if (tempC == DEVICE_DISCONNECTED_C) {
    return 22.0; // Fallback default
  }
  return tempC;
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    // Gather physical probe readings
    float ph = readPH();
    float turb = readTurbidity();
    float temp = readTemperature();

    // Construct JSON payload
    // Note: Heavy metals and lab BOD are set to safe ambient baseline if not equipped with ion-selective probes
    String payload = "{";
    payload += "\"location_id\":\"STATION_FIELD_ESP32_01\",";
    payload += "\"sensor_id\":\"ESP32_NODE_DEV\",";
    payload += "\"source_type\":\"iot_stream\",";
    payload += "\"features\":{";
    payload += "\"pH\":" + String(ph, 3) + ",";
    payload += "\"Turbidity\":" + String(turb, 2) + ",";
    payload += "\"Temperature\":" + String(temp, 2) + ",";
    payload += "\"Dissolved Oxygen\":7.5,";
    payload += "\"Biochemical Oxygen Demand\":1.2,";
    payload += "\"Lead\":0.002,";
    payload += "\"Mercury\":0.0001,";
    payload += "\"Arsenic\":0.002";
    payload += "},";
    payload += "\"confidence_level\":0.90";
    payload += "}";

    Serial.println("Transmitting telemetry packet:");
    Serial.println(payload);

    int httpResponseCode = http.POST(payload);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.print("HTTP Response Code: ");
      Serial.println(httpResponseCode);
      Serial.println("Server Response: " + response);
    } else {
      Serial.print("HTTP Error: ");
      Serial.println(httpResponseCode);
    }

    http.end();
  } else {
    Serial.println("Wi-Fi disconnected. Reconnecting...");
    WiFi.reconnect();
  }

  // Sample every 5 seconds
  delay(5000);
}
