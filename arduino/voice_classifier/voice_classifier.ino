#define BUFFER_SIZE 53   // Nombre de taps FIR
#define OUTPUT_SIZE 256  // Nombre d'échantillons filtrés par lot
#define DECIMATION 4     // 32kHz → 8kHz
#define HOP_SIZE 89

#include <Arduino.h>
#include "arduinoMFCC.h"
#include "model_weights.h"
#include <Adafruit_SSD1306.h>
#include <Adafruit_GFX.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET-1
#define OLED_ADDR 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT,&Wire, OLED_RESET);

volatile int16_t tempoBuffer[BUFFER_SIZE] = { 0 };
volatile uint8_t adcBufferIndex = 0;
volatile bool newSampleReady = false;
volatile uint8_t decimationCounter = 0;
volatile bool first = false;


void setupOLED() {
  if(!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println(F("SSD1306 allocation failed, check screen adress, or connection"));
    while(1) {}
  }

}

#define MFCC_SIZE 26
#define DCT_MFCC_SIZE 13
#define FRAME_SIZE 256
#define FREQ_ECH   8000
#define PAD 1

float mfcc_matrix[INPUT_H * INPUT_W];                   
float buffer_a   [INPUT_H * INPUT_W * CONV1_FILTERS];    
float buffer_b   [POOL_H * POOL_W * CONV1_FILTERS]; 
float dense1_out [DENSE1_UNITS];                         
float result     [NUM_CLASSES];                          

arduinoMFCC mymfcc(MFCC_SIZE,DCT_MFCC_SIZE, FRAME_SIZE, FREQ_ECH);

float filteredSignal[OUTPUT_SIZE];
float mfcc[MFCC_SIZE];
float mfccWithDCT[MFCC_SIZE];
float hop[HOP_SIZE];

static int filter_taps[BUFFER_SIZE] = {
  -708, -489, -398, -101, 338,
  777, 1052, 1039, 726, 231, -234, -456, -318,
  132, 682, 1037, 955, 378, -507, -1307, -1559,
  -925, 655, 2889, 5212, 6960, 7610, 6960, 5212,
  2889, 655, -925, -1559, -1307, -507, 378, 955,
  1037, 682, 132, -318, -456, -234, 231, 726,
  1039, 1052, 777, 338, -101, -398, -489, -708
};


void setupTimer() {
  PMC->PMC_PCER0 |= PMC_PCER0_PID27;
  TC0->TC_CHANNEL[0].TC_CMR = TC_CMR_TCCLKS_TIMER_CLOCK1 | TC_CMR_CPCTRG;
  TC0->TC_CHANNEL[0].TC_RC = 1312;
  TC0->TC_CHANNEL[0].TC_IER = TC_IER_CPCS;
  NVIC_EnableIRQ(TC0_IRQn);
  TC0->TC_CHANNEL[0].TC_CCR = TC_CCR_CLKEN | TC_CCR_SWTRG;
}

void TC0_Handler() {
  TC0->TC_CHANNEL[0].TC_SR;
  ADC->ADC_CR = ADC_CR_START;
}

void setupADC() {
  PMC->PMC_PCER1 |= PMC_PCER1_PID37;
  ADC->ADC_MR = ADC_MR_TRGEN_DIS
                | ADC_MR_PRESCAL(3)
                | ADC_MR_STARTUP_SUT64
                | ADC_MR_TRACKTIM(15)
                | ADC_MR_SETTLING_AST3;
  //ADC->ADC_MR |=ADC_MR_LOWRES;
  ADC->ADC_CHER |= ADC_CHER_CH7;
  ADC->ADC_IER = ADC_IER_EOC7;
  NVIC_EnableIRQ(ADC_IRQn);
}

void ADC_Handler() {
  if (!(ADC->ADC_ISR & ADC_ISR_EOC7)) return;  //vérifie si le flag activé est celui du channel 7

  float sample = ((float)ADC->ADC_CDR[7] - 1527);  // On enlève l’offset DC du micro (environ 1527) pour centrer le signal autour de 0,
                                                       // puis on applique un gain x2 (<<1) si nécessaire (je l'ai pas mis)
  tempoBuffer[adcBufferIndex] = sample;
  adcBufferIndex++;
  if (adcBufferIndex >= BUFFER_SIZE) adcBufferIndex = 0;  // Buffer circulaire : on revient au début du tableau quand on atteint la fin

  decimationCounter++;
  if (decimationCounter >= DECIMATION) {  //on prend 1 case de notre
    decimationCounter = 0;
    newSampleReady = true;
  }
}

void computeRIF() {
  if (first) {
    first = false;
    for (int outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex++) {  //on parcours toutes les cases du tableau

      while (!newSampleReady)
        ;  //1 sample sur 4 car on réduit notre fréquence de 16 à 8 kHz
      newSampleReady = false;

      uint8_t sumIndex = adcBufferIndex;

      int32_t acc = 0;
      for (int l = 0; l < BUFFER_SIZE; l++) {
        if (sumIndex == 0) sumIndex = BUFFER_SIZE;
        sumIndex--;
        acc += (int32_t)filter_taps[l] * (int32_t)tempoBuffer[sumIndex];  //convolution de notre entrée et de nos valeurs de sorties
      }

      filteredSignal[outputIndex] = ((float)(acc >> 15));  // Stocke le résultat filtré en ramenant à l'échelle correcte (normalisation Q15) (à cause de filter_taps)

      if (outputIndex >= (OUTPUT_SIZE - HOP_SIZE)) {
        hop[outputIndex - (OUTPUT_SIZE - HOP_SIZE)] = ((float)(acc >> 15));
      }
    }
  } else {
    for (int l = 0; l < HOP_SIZE; l++) {
      filteredSignal[l] = hop[l];
    }
    for (int m = 0; m < (OUTPUT_SIZE - HOP_SIZE); m++) {
      while (!newSampleReady);  //1 sample sur 4 car on réduit notre fréquence de 16 à 8 kHz
      newSampleReady = false;

      uint8_t sumIndex = adcBufferIndex;

      int32_t acc = 0;
      for (int j = 0; j < BUFFER_SIZE; j++) {
        if (sumIndex == 0) sumIndex = BUFFER_SIZE;
        sumIndex--;
        acc += (int32_t)filter_taps[j] * (int32_t)tempoBuffer[sumIndex];  //convolution de notre entrée et de nos valeurs de sorties
      }

      filteredSignal[m + HOP_SIZE] = ((float)(acc >> 15));
      if(m+HOP_SIZE>=(OUTPUT_SIZE-HOP_SIZE)){
        hop[m + HOP_SIZE - (OUTPUT_SIZE-HOP_SIZE)] = ((float)(acc >> 15));
      }
    }

  }
}



void convolution2D(const float* in, const float* w, const float* b, float* out, int in_h, int in_w, int in_c, int out_c, int kernel) {
    const int pad = kernel / 2;

    for (int k = 0; k < out_c; k++) {                  // pour chaque filtre de sortie
        for (int i = 0; i < in_w; i++) {               // colonnes
            for (int j = 0; j < in_h; j++) {           // lignes
                float sum = b[k];                       // on initialise avec le biais

                for (int kh = 0; kh < kernel; kh++) {
                    int h_in = j + kh - pad;
                    if (h_in < 0 || h_in >= in_h) continue;  // padding zero

                    for (int kw = 0; kw < kernel; kw++) {
                        int w_in = i + kw - pad;
                        if (w_in < 0 || w_in >= in_w) continue;  // padding zero

                        for (int c = 0; c < in_c; c++) {       // boucle sur les canaux d'entree
                            float in_val = in[(h_in * in_w + w_in) * in_c + c];
                            float w_val  = w[((kh * kernel + kw) * in_c + c) * out_c + k];
                            sum += in_val * w_val;
                        }
                    }
                }
                out[(j * in_w + i) * out_c + k] = (sum > 0.0f) ? sum : 0.0f;  // ReLU
            }
        }
    }
}

void maxpool2d_2x2(const float* in, float* out, int in_h, int in_w, int channels) {
    const int out_h = in_h / 2;
    const int out_w = in_w / 2;

    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            for (int c = 0; c < channels; c++) {
                int h_in = h * 2;
                int w_in = w * 2;

                float v00 = in[(h_in    ) * in_w * channels + (w_in    ) * channels + c];
                float v01 = in[(h_in    ) * in_w * channels + (w_in + 1) * channels + c];
                float v10 = in[(h_in + 1) * in_w * channels + (w_in    ) * channels + c];
                float v11 = in[(h_in + 1) * in_w * channels + (w_in + 1) * channels + c];

                float m = v00;
                if (v01 > m) m = v01;
                if (v10 > m) m = v10;
                if (v11 > m) m = v11;

                out[(h * out_w + w) * channels + c] = m;
            }
        }
    }
}

void dense(const float* in, const float* w, const float* b, float* out, int in_size, int out_size, bool apply_relu) {
    for (int j = 0; j < out_size; j++) {
        float sum = b[j];
        for (int i = 0; i < in_size; i++) {
            sum += in[i] * w[i * out_size + j];
        }
        out[j] = apply_relu ? ((sum > 0.0f) ? sum : 0.0f) : sum;
    }
}

// Softmax in-place sur un vecteur de taille n.
void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        total += x[i];
    }

    for (int i = 0; i < n; i++) {
        x[i] /= total;
    }
}

void normalize_zscore_per_coeff(float* x, int n_frames, int n_coeff) {
    for (int k = 0; k < n_coeff; k++) {
        float sum = 0.0f, sum2 = 0.0f;
        for (int i = 0; i < n_frames; i++) {
            float v = x[i * n_coeff + k];
            sum  += v;
            sum2 += v * v;
        }
        float mean = sum / n_frames;
        float var  = sum2 / n_frames - mean * mean;
        if (var < 0.0f) var = 0.0f;            // garde-fou arrondi
        float stdv = sqrtf(var) + 1e-6f;       // epsilon APRES sqrt, comme en Python
        for (int i = 0; i < n_frames; i++)
            x[i * n_coeff + k] = (x[i * n_coeff + k] - mean) / stdv;
    }
}

void CNN(const float* in) {
  // Conv1 : (INPUT_H, INPUT_W, INPUT_C) -> (INPUT_H, INPUT_W, CONV1_FILTERS)
  convolution2D(in, conv1_w, conv1_b, buffer_a,
                INPUT_H, INPUT_W, INPUT_C, CONV1_FILTERS, KERNEL_SIZE);

  // MaxPool 1 : (INPUT_H, INPUT_W, CONV1_FILTERS) -> (POOL_H, POOL_W, CONV1_FILTERS)
  maxpool2d_2x2(buffer_a, buffer_b,
                INPUT_H, INPUT_W, CONV1_FILTERS);

  // Conv2 : (POOL_H, POOL_W, CONV1_FILTERS) -> (POOL_H, POOL_W, CONV2_FILTERS)
  convolution2D(buffer_b, conv2_w, conv2_b, buffer_a,
                POOL_H, POOL_W, CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE);

  // MaxPool 2 : (POOL_H, POOL_W, CONV2_FILTERS) -> (POOL2_H, POOL2_W, CONV2_FILTERS)
  maxpool2d_2x2(buffer_a, buffer_b,
                POOL_H, POOL_W, CONV2_FILTERS);

  // Dense1 + ReLU : FLAT_SIZE -> DENSE1_UNITS (lit buffer_b, la sortie du 2e pool)
  dense(buffer_b, dense1_w, dense1_b, dense1_out,
        FLAT_SIZE, DENSE1_UNITS, true);

  // Dense2 (logits) : DENSE1_UNITS -> NUM_CLASSES
  dense(dense1_out, dense2_w, dense2_b, result,
        DENSE1_UNITS, NUM_CLASSES, false);

  // Softmax sur les logits de sortie
  softmax(result, NUM_CLASSES);
}

void setup() {
  Serial.begin(460800);
  while (!Serial)
    ;
  setupTimer();
  setupADC();
  pinMode(52, INPUT_PULLUP);
  mymfcc.create_hamming_window();
  mymfcc.create_mel_filter_bank(); 
  mymfcc.create_dct_matrix();
  setupOLED();
}

void loop() {
  bool c = digitalRead(52);
  if (c == false) {
    display.clearDisplay();

    for (int i = 0; i < 48; i++) {
      if (i == 0) first = true;
      computeRIF();
      mymfcc.computeWithDCT(filteredSignal, mfccWithDCT);
      for (int k = 0; k < INPUT_W; k++) {
        mfcc_matrix[i * INPUT_W + k] = mfccWithDCT[k];  // copie des frames dans mfcc_matrix
      }
    }

    normalize_zscore_per_coeff(mfcc_matrix, INPUT_H, INPUT_W);
    CNN(mfcc_matrix);

    if (result[0] > 0.5f) {

      display.setCursor(30, 32);
      display.setTextColor(WHITE);
      display.setTextSize(1);
      display.print("Hello : ");
      display.print(result[0]);
      display.display();
      // Serial.print("Hello ");
      // Serial.println(result[0]);
    } else if (result[1] > 0.5f) {

      display.setCursor(17, 32);
      display.setTextColor(WHITE);
      display.setTextSize(1);
      display.print("Thank you : ");
      display.print(result[1]);
      display.display();
      // Serial.print("Thank you ");
      // Serial.println(result[1]);
    }
  }
}







