/*
  ============================================================================
  VibraSense AI — Final Multimodal Demo Firmware v3.1
  ============================================================================
  Features: 13 (Vibration RMS/Peak, FFT, Audio RMS/Peak/ZCR, RPM, Temp, FIFO)
  Logic: Multimodal Fusion (Vib * 0.6 + Audio * 0.4)
  ============================================================================
*/

#include <Arduino.h>
#include <Wire.h>
#include <U8g2lib.h>
#include <ESP_I2S.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <arduinoFFT.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// TRAINED EDGE MODEL PARAMETERS
// ============================================================================
#define VIBRASENSE_EDGE_MODEL_READY    1
#define VIBRASENSE_EDGE_FEATURE_COUNT  13

static const float VIBRASENSE_EDGE_MEDIAN[VIBRASENSE_EDGE_FEATURE_COUNT] = {
  0.1668f, 0.3098f, 0.31475f, 0.38925f, 0.33645f, 0.32055f, 7.3f, 
  429565.8f, 1173278.0f, 0.032f, 0.0f, 29.0f, 1.0f
};

static const float VIBRASENSE_EDGE_SCALE[VIBRASENSE_EDGE_FEATURE_COUNT] = {
  0.1f, 0.1549f, 0.157375f, 0.40749261f, 0.168225f, 0.160275f, 10.0f, 
  374770.442f, 1026464.77f, 1000.0f, 50.0f, 1.0f, 1.0f
};
// ============================================================================

static const int SDA_PIN=8, SCL_PIN=9, ADXL_ADDR=0x53, I2S_BCLK=4, I2S_WS=5, I2S_SD=6, HALL_PIN=7, ONEWIRE_PIN=15;
static const int RED_LED_PIN=21, GREEN_LED_PIN=47, YELLOW_LED_PIN=48;

static const uint8_t FIFO_WATERMARK = 16;
static const uint8_t FIFO_STREAM    = 0x80 | FIFO_WATERMARK;
static const uint8_t REG_DEVID=0x00, REG_BW_RATE=0x2C, REG_POWER_CTL=0x2D, REG_INT_SOURCE=0x30, REG_DATA_FORMAT=0x31, REG_DATAX0=0x32, REG_FIFO_CTL=0x38, REG_FIFO_STATUS=0x39;
static const uint32_t HALL_DEBOUNCE_US = 2000;

I2SClass I2S;
OneWire oneWire(ONEWIRE_PIN);
DallasTemperature tempSensor(&oneWire);
U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);

// FFT Setup (128 samples at 200Hz)
const uint16_t SAMPLES = 128;
double vReal[SAMPLES];
double vImag[SAMPLES];
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, SAMPLES, 200.0);

bool adxlReady=false, audioReady=false, oledReady=false;
uint32_t i2cErrors=0, fifoOverruns=0, maxFifoDepth=0, windowId=0, windowStartMs=0;
volatile uint32_t hallPulsesTotal=0, hallLastPulseUs=0, hallPulsesAtWindowStart=0;
bool tempConversionPending=false; uint32_t tempRequestMs=0; float latestTempC=NAN; bool temperatureValid=false;

uint32_t accelSamples=0;
float accelVals[3][128];
double accelSum[3]={0,0,0}, accelSumSq[3]={0,0,0};
float accelMin[3]={1e9f,1e9f,1e9f}, accelMax[3]={-1e9f,-1e9f,-1e9f};

uint64_t audioSamples=0; double audioSumSq=0; int32_t audioPeak=0, audioLast=0; uint32_t audioZCR=0;

static int32_t i2sBuffer[512];

void IRAM_ATTR hallISR() {
  uint32_t now = (uint32_t)micros();
  if ((uint32_t)(now - hallLastPulseUs) >= HALL_DEBOUNCE_US) {
    hallPulsesTotal++; hallLastPulseUs = now;
  }
}

bool writeReg(uint8_t r, uint8_t v) {
  Wire.beginTransmission(ADXL_ADDR); Wire.write(r); Wire.write(v);
  if (Wire.endTransmission() != 0) { i2cErrors++; return false; } return true;
}
bool readRegs(uint8_t r, uint8_t *o, size_t n) {
  Wire.beginTransmission(ADXL_ADDR); Wire.write(r);
  if (Wire.endTransmission(false) != 0) { i2cErrors++; return false; }
  if (Wire.requestFrom((int)ADXL_ADDR, (int)n, (int)true) != n) { i2cErrors++; return false; }
  for (size_t i=0; i<n; i++) o[i] = (uint8_t)Wire.read();
  return true;
}
bool configureADXL345() {
  uint8_t id=0; if (!readRegs(REG_DEVID, &id, 1) || id != 0xE5) return false;
  writeReg(REG_POWER_CTL,0x00); writeReg(REG_FIFO_CTL,0x00);
  writeReg(REG_DATA_FORMAT,0x08); writeReg(REG_BW_RATE,0x0B);
  writeReg(REG_FIFO_CTL,FIFO_STREAM); writeReg(REG_POWER_CTL,0x08);
  return true;
}

void addAccelSample(int16_t raw[3]) {
  if (accelSamples < 128) {
    for (int a=0; a<3; a++) {
      float g = raw[a] * 0.0039f;
      accelVals[a][accelSamples] = g;
      accelSum[a]+=g; accelSumSq[a]+=(double)g*g;
      if (g<accelMin[a]) accelMin[a]=g;
      if (g>accelMax[a]) accelMax[a]=g;
    }
  }
  accelSamples++;
}

void serviceADXL345() {
  uint8_t is=0; if (!readRegs(REG_INT_SOURCE, &is, 1)) return;
  if (is & 0x01) fifoOverruns++;
  uint8_t fs=0; if (!readRegs(REG_FIFO_STATUS, &fs, 1)) return;
  uint8_t entries = fs & 0x3F;
  if (entries > maxFifoDepth) maxFifoDepth = entries;
  for (uint8_t i=0; i<entries; i++) {
    uint8_t b[6]; if (!readRegs(REG_DATAX0, b, 6)) return;
    int16_t raw[3] = { (int16_t)((uint16_t)b[1]<<8|b[0]), (int16_t)((uint16_t)b[3]<<8|b[2]), (int16_t)((uint16_t)b[5]<<8|b[4]) };
    addAccelSample(raw);
  }
}

void serviceAudio() {
  if (!audioReady) return;
  int avail = I2S.available(); if (avail <= 0) return;
  int chunk = avail > (int)sizeof(i2sBuffer) ? (int)sizeof(i2sBuffer) : avail;
  chunk -= chunk % 8;
  size_t got = I2S.readBytes((char *)i2sBuffer, (size_t)chunk);
  const size_t frames = got / 8;
  for (size_t i=0; i<frames; i++) {
    int32_t s = i2sBuffer[i*2] >> 8;
    if (s > audioPeak) audioPeak = s;
    if ((s >= 0 && audioLast < 0) || (s < 0 && audioLast >= 0)) audioZCR++;
    audioLast = s;
    audioSumSq += (double)s * (double)s;
    audioSamples++;
  }
}

void serviceTemperature() {
  if (tempConversionPending && (uint32_t)(millis() - tempRequestMs) >= 400) {
    float t = tempSensor.getTempCByIndex(0);
    if (t > -55.0f && t < 125.0f && t != DEVICE_DISCONNECTED_C) {
      latestTempC = t; temperatureValid = true;
    } else { temperatureValid = false; }
    tempSensor.requestTemperatures();
    tempRequestMs = millis(); tempConversionPending = true;
  }
}

void resetWindowStats() {
  accelSamples=0;
  for(int a=0; a<3; a++){ accelSum[a]=0; accelSumSq[a]=0; accelMin[a]=1e9f; accelMax[a]=-1e9f; }
  audioSamples=0; audioSumSq=0; audioPeak=0; audioLast=0; audioZCR=0;
  noInterrupts(); hallPulsesAtWindowStart = hallPulsesTotal; interrupts();
}

void emitWindow() {
  uint32_t now = millis();
  float elapsed = (now - windowStartMs) / 1000.0f;
  if (elapsed <= 0.0f) elapsed = 1.0f;
  uint32_t pulses = hallPulsesTotal - hallPulsesAtWindowStart;

  float mean[3], rms[3], peak[3];
  if (accelSamples > 0) {
    for (int a=0; a<3; a++) {
      mean[a] = accelSum[a] / accelSamples;
      rms[a]  = sqrtf(fmaxf(0, (float)(accelSumSq[a] / accelSamples) - mean[a]*mean[a]));
      peak[a] = fmaxf(fabsf(accelMin[a]-mean[a]), fabsf(accelMax[a]-mean[a]));
    }
  } else { for(int a=0; a<3; a++){ rms[a]=0; peak[a]=0; } }

  // FFT Calculation for Vibration
  float fftPeak = 0.0f;
  if (accelSamples >= 128) {
    for (int i=0; i<128; i++) { vReal[i] = accelVals[0][i]; vImag[i] = 0; }
    FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
    FFT.compute(FFTDirection::Forward);
    FFT.complexToMagnitude();
    fftPeak = (float)FFT.majorPeak();
  }

  float audioRms = audioSamples ? sqrtf((float)(audioSumSq/audioSamples)) : 0.0f;
  float audioZcrRate = audioSamples ? (float)audioZCR / audioSamples : 0.0f;
  float rpm = pulses * 60.0f / elapsed;
  float tempOut = temperatureValid ? latestTempC : 25.0f;

  // 13-Feature Vector
  float fv[13] = {
    rms[0], rms[1], rms[2],
    peak[0], peak[1], peak[2],
    fftPeak,
    audioRms, (float)audioPeak, audioZcrRate,
    rpm, tempOut, (float)fifoOverruns
  };

  // Multimodal Fusion Logic
  float zVib=0, zAud=0, zCtx=0;
  for (int i=0; i<13; i++) {
    float s = VIBRASENSE_EDGE_SCALE[i]; if (s < 1e-9f) s = 1e-9f;
    float z = fabsf(fv[i] - VIBRASENSE_EDGE_MEDIAN[i]) / s;
    if (i < 7)      { if (z > zVib) zVib = z; }
    else if (i < 10){ if (z > zAud) zAud = z; }
    else            { if (z > zCtx) zCtx = z; }
  }
  
  float fused = zVib*0.6f + zAud*0.4f;
  const char *status;
  int ledRed = 0, ledYellow = 0;

  // 3-Tier Thresholds
  if ((zVib > 4.0f && zAud > 4.0f) || fused > 4.5f || zCtx > 6.0f) {
    status = "ANOMALY_PERSISTENT";
    ledRed = 1;
  } else if (fused > 2.5f) {
    status = "ANOMALY_CANDIDATE";
    ledYellow = 1;
  } else {
    status = "NORMAL_PATTERN";
  }

  // Update LEDs
  digitalWrite(RED_LED_PIN, ledRed ? HIGH : LOW);
  digitalWrite(YELLOW_LED_PIN, ledYellow ? HIGH : LOW);
  digitalWrite(GREEN_LED_PIN, (!ledRed && !ledYellow) ? HIGH : LOW);

  // Update OLED
  if (oledReady) {
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 10, "VibraSense Edge AI");
    u8g2.drawLine(0, 13, 128, 13);

    if (ledRed) {
      u8g2.setFont(u8g2_font_helvB14_tr);
      u8g2.drawStr(14, 34, "ANOMALY");
    } else if (ledYellow) {
      u8g2.setFont(u8g2_font_helvB12_tf);
      u8g2.drawStr(20, 32, "WARNING");
    } else {
      u8g2.setFont(u8g2_font_helvB12_tf);
      u8g2.drawStr(24, 32, "HEALTHY");
    }

    u8g2.setFont(u8g2_font_6x10_tf);
    char buf[32];
    snprintf(buf, sizeof(buf), "Fused: %.2f", fused);
    u8g2.drawStr(0, 46, buf);
    snprintf(buf, sizeof(buf), "VibZ:%.1f AudZ:%.1f", zVib, zAud);
    u8g2.drawStr(0, 56, buf);
    u8g2.sendBuffer();
  }

  // Print CSV Data to Serial
  Serial.printf("DATA,demo_session,%lu,%0.3f,demo,1,housing,%lu,200,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.1f,%llu,16000,%0.1f,%ld,%0.3f,%lu,%0.1f,%0.1f,%d,%lu,%lu\n",
    windowId, elapsed, accelSamples,
    rms[0], rms[1], rms[2],
    peak[0], peak[1], peak[2],
    fftPeak,
    audioSamples, audioRms, audioPeak, audioZcrRate,
    pulses, rpm, tempOut, temperatureValid?1:0,
    i2cErrors, fifoOverruns);

  windowId++;
  windowStartMs = now;
  resetWindowStats();
}

void setup() {
  Serial.begin(115200); delay(1200);
  pinMode(RED_LED_PIN, OUTPUT); pinMode(GREEN_LED_PIN, OUTPUT); pinMode(YELLOW_LED_PIN, OUTPUT);
  digitalWrite(GREEN_LED_PIN, HIGH);
  
  Wire.begin(SDA_PIN, SCL_PIN, 100000); Wire.setTimeOut(10);
  u8g2.begin(); oledReady = true;
  
  adxlReady = configureADXL345();
  pinMode(HALL_PIN, INPUT); attachInterrupt(digitalPinToInterrupt(HALL_PIN), hallISR, FALLING);
  
  tempSensor.begin(); tempSensor.setWaitForConversion(false); tempSensor.setResolution(11);
  tempSensor.requestTemperatures(); tempRequestMs = millis(); tempConversionPending = true;
  
  I2S.setPins(I2S_BCLK, I2S_WS, -1, I2S_SD);
  audioReady = I2S.begin(I2S_MODE_STD, 16000, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO);
  if (audioReady) I2S.setTimeout(50);

  windowStartMs = millis();
  resetWindowStats();
}

void loop() {
  serviceTemperature();
  serviceAudio();
  serviceADXL345();
  if ((uint32_t)(millis() - windowStartMs) >= 1000) emitWindow();
  delay(1);
}