// #include <WiFi.h>
// #include <Firebase_ESP_Client.h>
// #include "addons/TokenHelper.h"
// #include "addons/RTDBHelper.h"
// #include "DHT.h"

// // ================== WiFi Credentials ==================
// #define WIFI_SSID "Airtel_praj_8505"
// #define WIFI_PASSWORD "Air@30281"

// // ================== Firebase Credentials ==================
// #define API_KEY "AIzaSyBsiQZIwtAyQcBNJMUWziJ6mwCTCV3SWCk"
// #define DATABASE_URL "https://agri-hill-default-rtdb.asia-southeast1.firebasedatabase.app"

// // ================== Pins ==================
// #define DHTPIN 21       // DHT22 DATA pin
// #define DHTTYPE DHT22
// #define SOIL_PIN 35     // Soil moisture sensor (ADC)
// #define WATER_PIN 32    // Water level sensor (ADC)
// #define RELAY_PIN 26    // Relay module (pump)

// // ================== Objects ==================
// DHT dht(DHTPIN, DHTTYPE);
// FirebaseData fbdo;
// FirebaseAuth auth;
// FirebaseConfig config;

// // ================== Setup ==================
// void setup() {
//   Serial.begin(115200);

//   pinMode(RELAY_PIN, OUTPUT);
//   digitalWrite(RELAY_PIN, HIGH); // Relay OFF initially (active LOW)

//   dht.begin();

//   // WiFi connect
//   WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
//   Serial.print("Connecting to Wi-Fi");
//   while (WiFi.status() != WL_CONNECTED) {
//     Serial.print(".");
//     delay(300);
//   }
//   Serial.println("\n✅ Connected with IP: " + WiFi.localIP().toString());

//   // Firebase config
//   config.api_key = API_KEY;
//   config.database_url = DATABASE_URL;

//   if (Firebase.signUp(&config, &auth, "", "")) {
//     Serial.println("✅ Firebase signup successful");
//   } else {
//     Serial.printf("❌ Signup error: %s\n", config.signer.signupError.message.c_str());
//   }

//   Firebase.begin(&config, &auth);
//   Firebase.reconnectWiFi(true);
// }

// // ================== Loop ==================
// void loop() {
//   // 🌡 DHT22 readings
//   float humidity = dht.readHumidity();
//   float temperature = dht.readTemperature();

//   if (isnan(humidity) || isnan(temperature)) {
//     Serial.println("❌ Failed to read from DHT sensor!");
//   } else {
//     Serial.printf("🌡 Temp: %.2f °C | 💧 Hum: %.2f %%\n", temperature, humidity);
//   }

//   // 🌱 Soil moisture (in %)
//   int soilRaw = analogRead(SOIL_PIN);
//   int soilPercent = map(soilRaw, 4095, 1000, 0, 100); // adjust calibration
//   soilPercent = constrain(soilPercent, 0, 100);
//   Serial.printf("🌱 Soil Moisture: %d %% (Raw: %d)\n", soilPercent, soilRaw);

//   // 💧 Water level (in %)
//   int waterRaw = analogRead(WATER_PIN);
//   int waterPercent = map(waterRaw, 950, 3500, 0, 100); // adjust calibration
//   waterPercent = constrain(waterPercent, 0, 100);
//   Serial.printf("🛑 Water Level: %d %% (Raw: %d)\n", waterPercent, waterRaw);

//   // 🚰 Pump control logic
//   bool pumpState = false;
//   bool manualOverride = false;

//   if (Firebase.ready()) {
//     // Read manual override flag from Firebase
//     if (Firebase.RTDB.getBool(&fbdo, "/actuators/manualOverride")) {
//       manualOverride = fbdo.boolData();
//     }

//     // Read pump command from Firebase (only if manualOverride is true)
//     if (manualOverride) {
//       if (Firebase.RTDB.getBool(&fbdo, "/actuators/pump")) {
//         pumpState = fbdo.boolData();
//         digitalWrite(RELAY_PIN, pumpState ? LOW : HIGH);
//         Serial.printf("🔘 Manual Pump Control: %s\n", pumpState ? "ON" : "OFF");
//       }
//     } else {
//       // Automatic mode
//       if (soilPercent < 25 && waterPercent > 5) {
//         digitalWrite(RELAY_PIN, LOW);   // Pump ON
//         pumpState = true;
//         Serial.println("🚰 Auto Pump: ON");
//       } else {
//         digitalWrite(RELAY_PIN, HIGH);  // Pump OFF
//         pumpState = false;
//         Serial.println("💡 Auto Pump: OFF");
//       }
//     }

//     // 🔗 Send sensor data + pump state to Firebase
//     Firebase.RTDB.setFloat(&fbdo, "/sensors/temperature", temperature);
//     Firebase.RTDB.setFloat(&fbdo, "/sensors/humidity", humidity);
//     Firebase.RTDB.setInt(&fbdo, "/sensors/soilMoisturePercent", soilPercent);
//     Firebase.RTDB.setInt(&fbdo, "/sensors/waterLevelPercent", waterPercent);
//     Firebase.RTDB.setBool(&fbdo, "/actuators/pump", pumpState);
//   }

//   delay(2000); // every 2s
// }
