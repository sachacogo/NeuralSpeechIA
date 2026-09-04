#define BUFFER_SIZE 53   // Nombre de taps FIR
#define OUTPUT_SIZE 256  // Nombre d'échantillons filtrés par lot
#define DECIMATION 4     // 32kHz → 8kHz
#define HOP_SIZE 89

#include <Arduino.h>
#include "arduinoMFCC.h"

volatile int16_t tempoBuffer[BUFFER_SIZE] = { 0 };
volatile uint8_t adcBufferIndex = 0;
volatile bool newSampleReady = false;
volatile uint8_t decimationCounter = 0;
volatile bool first = false;


#define MFCC_SIZE 26
#define DCT_MFCC_SIZE 13
#define FRAME_SIZE 256
#define FREQ_ECH   8000

arduinoMFCC mymfcc(MFCC_SIZE,DCT_MFCC_SIZE, FRAME_SIZE, FREQ_ECH);

// int16_t filteredSignal[OUTPUT_SIZE];
float filteredSignal[OUTPUT_SIZE]; //on utilise un tableau de float pour la bibliothèque MFCC
float mfcc[MFCC_SIZE];
float mfccWithDCT[MFCC_SIZE];
float hop[HOP_SIZE];

static int filter_taps[BUFFER_SIZE] = {
  -708,
  -489, -398, -101, 338,
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



void sendFrame() {
  Serial.write(0xAA);
  Serial.write(0x55);  //mot de syncrhonisation pour pas que notre audio soit corrompu
  Serial.write(0xAA);  
  Serial.write(0x55);  


  //data
  // Serial.write((uint8_t*)filteredSignal, OUTPUT_SIZE * sizeof(float));  //avec float pour la fft (pour ne pas perdre les valeurs >32k)
  Serial.write((uint8_t*)mfccWithDCT, DCT_MFCC_SIZE * sizeof(float));  //envoie coef mfcc
  // Serial.write((uint8_t*)mfcc, MFCC_SIZE * sizeof(float));  //envoie coef mfcc DCT


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
}

void loop() {
  bool c = digitalRead(52);
  if (c == false) {
    for (int i = 0; i < 48; i++) {  
      if (i == 0) {
        first = true;
      }
      computeRIF();
      // mymfcc.compute(filteredSignal, mfcc);
      mymfcc.computeWithDCT(filteredSignal, mfccWithDCT);
      sendFrame();
    }
  }
}





// void loop() {
//   bool c = digitalRead(52);
//   if (c == false) {
//     for (int i = 0; i < 48; i++) {  
//       if (i == 0) {
//         first = true;
//       }
//       computeRIF();
//       mymfcc.compute(filteredSignal, mfcc);
//       sendFrame();
//     }
//   }
// }


//loop pour le pré-emphase

// void loop() {
//   bool c = digitalRead(52);
//   if (c == false) {
//     for (int i = 0; i < 48; i++) {  
//       if (i == 0) {
//         first = true;
//       }
//       computeRIF();
//       // mymfcc.compute(filteredSignal, mfcc);
//       for(int i = 0; i < OUTPUT_SIZE; i++){
//         mymfcc._frame[i] = (float)filteredSignal[i];
//       }

//       mymfcc.pre_emphasis();
//       mymfcc.apply_hamming_window();
//       mymfcc.apply_fft();

//       // for(int i = 0; i < OUTPUT_SIZE; i++){
//       //   filteredSignal[i] = (int16_t)mymfcc._frame[i];
//       // }

//       sendFrame();
//     }
//   }
// }