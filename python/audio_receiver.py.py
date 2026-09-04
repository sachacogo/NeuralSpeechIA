import serial
import time

PORT        = 'COM5'
BAUD        = 460800
SAMPLE_RATE = 8000
OUTPUT_SIZE = 256             # échantillons par trame (int16_t)
FRAME_BYTES = OUTPUT_SIZE * 2 # octets par trame (2 pour int16)
SYNC        = bytes([0xAA, 0x55])

ser = serial.Serial(PORT, BAUD, timeout=5)
time.sleep(2)
print("Connecté à", PORT)


def attendre_sync(ser):
    """Cherche le mot de sync 0xAA55 dans le flux binaire."""
    buf = b''
    while True:
        octet = ser.read(1)
        if not octet:
            print("Timeout — pas de réponse Arduino")
            return False
        buf += octet
        if buf[-2:] == SYNC:
            return True

def lire_trame(ser):
    """Lit une trame complète après synchronisation."""
    if not attendre_sync(ser):
        return None
    payload = ser.read(FRAME_BYTES)
    if len(payload) != FRAME_BYTES:
        print(f"Trame incomplète ({len(payload)}/{FRAME_BYTES})")
        return None
    return payload


audio_data = bytearray()

print("Enregistrement... Ctrl+C pour arrêter")
try:
    while True:
        trame = lire_trame(ser)
        if trame:
            audio_data += trame

except KeyboardInterrupt:
    pass

ser.close()

with open('elec.bin', 'wb') as f:
    f.write(bytes(audio_data))

print(f"Sauvegardé → elec.bin  ({len(audio_data) // 2} échantillons, {len(audio_data) / (SAMPLE_RATE * 2):.1f}s)") # *2 //2 si int16 4 pour float