import serial
import time
import struct
import re
import sys
import os
import argparse

try:
    import msvcrt  # Windows : lecture clavier sans Entree
    def get_key():
        return msvcrt.getch().decode("utf-8", "ignore").lower()
except ImportError:
    def get_key():
        return (input() or "\r").lower()[:1]

import numpy as np

# ==================== CONFIG FIXE ====================
PORT        = 'COM5'
BAUD        = 460800
MFCC_SIZE   = 13
NB_FRAMES   = 48
FRAME_BYTES = MFCC_SIZE * 4
SYNC        = bytes([0xAA, 0x55, 0xAA, 0x55])


# ==================== PROTOCOLE SERIE ====================
def attendre_sync(ser):
    buf = b''
    while True:
        octet = ser.read(1)
        if not octet:
            continue
        buf += octet
        if buf[-4:] == SYNC:
            return


def lire_trame(ser):
    attendre_sync(ser)
    payload = b''
    while len(payload) < FRAME_BYTES:
        chunk = ser.read(FRAME_BYTES - len(payload))
        if chunk:
            payload += chunk
    return struct.unpack('<' + 'f' * MFCC_SIZE, payload)


def lire_sample(ser):
    sample = []
    for _ in range(NB_FRAMES):
        sample.append(lire_trame(ser))
    return sample


# ==================== NORMALISATION (identique) ====================
def normaliser_sample(sample):
    """Z-score par coefficient : chaque colonne (C0...C12) centree-reduite."""
    arr = np.array(sample)              # shape (48, 13)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0) + 1e-6
    return ((arr - mean) / std).tolist()


# ==================== I/O FICHIER C ====================
def sauvegarder(samples, fichier, nom_var):
    n = len(samples)
    with open(fichier, 'w') as f:
        f.write(f"const float {nom_var}[{n}][{NB_FRAMES}][{MFCC_SIZE}] =\n{{\n")
        for s_idx, sample in enumerate(samples):
            f.write("  {\n")
            for r_idx, row in enumerate(sample):
                values = ", ".join(f"{v:.3f}" for v in row)
                virgule = "," if r_idx < NB_FRAMES - 1 else ""
                f.write(f"    {{{values}}}{virgule}\n")
            virgule = "," if s_idx < n - 1 else ""
            f.write(f"  }}{virgule}\n")
        f.write("};\n")


def charger_existant(fichier):
    """Reparse un fichier C deja genere -> liste de samples (pour --append)."""
    if not os.path.exists(fichier):
        return []
    with open(fichier, 'r') as f:
        txt = f.read()
    nums = re.findall(r'-?\d+\.\d+(?:e[+-]?\d+)?', txt, re.IGNORECASE)
    vals = np.array([float(x) for x in nums], dtype=float)
    bloc = NB_FRAMES * MFCC_SIZE
    if vals.size == 0 or vals.size % bloc != 0:
        print(f"  /!\\ '{fichier}' illisible ou incomplet, on repart de zero.")
        return []
    arr = vals.reshape(-1, NB_FRAMES, MFCC_SIZE)
    return [s.tolist() for s in arr]


# ==================== BOUCLE PRINCIPALE ====================
def main():
    p = argparse.ArgumentParser(description="Capture MFCC depuis l'Arduino Due.")
    p.add_argument("var_name", help="nom de la variable C (ex: a, b, datatest)")
    p.add_argument("--out", help="fichier de sortie (defaut: <var_name>.txt)")
    p.add_argument("--n", type=int, default=0,
                   help="nombre de samples a capturer (0 = illimite, 'q' pour arreter)")
    p.add_argument("--append", action="store_true",
                   help="ajoute aux samples deja presents dans le fichier de sortie")
    p.add_argument("--port", default=PORT, help=f"port serie (defaut {PORT})")
    args = p.parse_args()

    fichier = args.out or f"{args.var_name}.txt"

    samples = charger_existant(fichier) if args.append else []
    deja = len(samples)
    if deja:
        print(f"Reprise : {deja} samples deja dans '{fichier}'.")

    ser = serial.Serial(args.port, BAUD, timeout=1)
    time.sleep(2)
    print(f"Connecte a {args.port} -> '{fichier}' (variable '{args.var_name}')")
    objectif = f"{args.n}" if args.n else "illimite"
    print(f"Objectif : {objectif} samples.")
    print("Apres chaque sample : [Entree]=garder  [r]=refaire  [q]=quitter+sauver\n")

    try:
        while True:
            if args.n and len(samples) - deja >= args.n:
                print("Objectif atteint.")
                break

            idx = len(samples) + 1
            print(f"  [Sample {idx}] Appuie sur le bouton et prononce le mot...")
            sample = normaliser_sample(lire_sample(ser))

            # Vide les rafales en trop si le bouton a ete maintenu trop longtemps
            ser.reset_input_buffer()

            print("  -> recu. [Entree]=garder  [r]=refaire  [q]=quitter : ", end="", flush=True)
            k = get_key()
            print()

            if k == 'q':
                break
            if k == 'r':
                print("  Sample ignore, recommence.\n")
                continue

            samples.append(sample)
            sauvegarder(samples, fichier, args.var_name)
            print(f"  Garde ({len(samples) - deja} nouveaux / {len(samples)} total)\n")

    except KeyboardInterrupt:
        print("\nInterrompu (Ctrl+C).")
    finally:
        ser.close()
        sauvegarder(samples, fichier, args.var_name)
        print(f"\nSauvegarde -> {fichier}  ({len(samples)} x {NB_FRAMES} x {MFCC_SIZE})")


if __name__ == "__main__":
    main()
