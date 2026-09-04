#define BUFFER_SIZE  53              // Nombre de taps FIR
#define OUTPUT_SIZE  256            // Nombre d'échantillons filtrés par lot
#define DECIMATION    4              // 32kHz → 8kHz

volatile int16_t tempoBuffer[BUFFER_SIZE] = {0};
volatile uint8_t adcBufferIndex           = 0;
volatile bool    newSampleReady           = false;
volatile uint8_t decimationCounter        = 0;

int16_t filteredSignal[OUTPUT_SIZE];


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
    TC0->TC_CHANNEL[0].TC_RC  = 1312;
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
    ADC->ADC_IER   = ADC_IER_EOC7;
    NVIC_EnableIRQ(ADC_IRQn);
}

void ADC_Handler() {
    if (!(ADC->ADC_ISR & ADC_ISR_EOC7)) return; //vérifie si le flag activé est celui du channel 7

    int16_t sample = ((int16_t)ADC->ADC_CDR[7]-1527); // On enlève l’offset DC du micro (environ 1527) pour centrer le signal autour de 0,
// puis on applique un gain x2 (<<1) si nécessaire (je l'ai pas mis)
    tempoBuffer[adcBufferIndex] = sample;
    adcBufferIndex++;
    if (adcBufferIndex >= BUFFER_SIZE) adcBufferIndex = 0; // Buffer circulaire : on revient au début du tableau quand on atteint la fin

    decimationCounter++; 
    if (decimationCounter >= DECIMATION) { //on prend 1 case de notre 
        decimationCounter = 0;
        newSampleReady = true;
    }
}

void computeRIF() {
    for (int outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex++) { //on parcours toutes les cases du tableau

        while (!newSampleReady); //1 sample sur 4 car on réduit notre fréquence de 16 à 8 kHz
        newSampleReady = false; 

        uint8_t sumIndex = adcBufferIndex;

        // int t0 = micros();

        int32_t acc = 0;
        for (int l = 0; l < BUFFER_SIZE; l++) { 
            if (sumIndex == 0) sumIndex = BUFFER_SIZE;
            sumIndex--;
            acc += (int32_t)filter_taps[l] * (int32_t)tempoBuffer[sumIndex]; //convolution de notre entrée et de nos valeurs de sorties
        }

        // int t1 = micros() - t0;

        filteredSignal[outputIndex] = ((int16_t)(acc >> 15)); // Stocke le résultat filtré en ramenant à l'échelle correcte (normalisation Q15) (à cause de filter_taps)
        // Serial.print("valeur signal ");
        // Serial.print(filteredSignal[outputIndex]);
        // Serial.print(" temps de conversion : ");
        // Serial.print(t1);
        // Serial.println(" µs");
    }
}


void sendFrame() {
  //Sync word
  Serial.write(0xAA);
  Serial.write(0x55); //mot de syncrhonisation pour pas que notre audio soit corrompu 

  //data
  Serial.write((uint8_t*)filteredSignal, OUTPUT_SIZE * sizeof(int16_t)); //envoie de 8 bits en 8 bits notre tableau de sortie vers notre COM

}

void setup() {
    Serial.begin(460800);
    while (!Serial);
    setupTimer();
    setupADC();
    pinMode(52,INPUT_PULLUP);
}

void loop() {
    bool c = digitalRead(52);
    if(c == false){
        for(int i = 0; i<32; i++){//256 samples, 8000 samples / secondes, -> 0.032 secondes par sample -> 32*0.032 environ = 1seconde
            computeRIF();
            sendFrame(); 

        }
    }
}


// loop débug entrée micro max
// void loop() {
//     bool c = digitalRead(52);
//     if(c == true){
//         int16_t maxVal = 0;
//         for(int i = 0; i < 32; i++){
//             computeRIF();
//             for(int j = 0; j < OUTPUT_SIZE; j++){
//                 int16_t absVal = abs(filteredSignal[j]);
//                 if(absVal > maxVal) maxVal = absVal;
//             }
//         }
//         Serial.println(maxVal);
//     }
// }