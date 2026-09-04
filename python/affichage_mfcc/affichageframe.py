import numpy as np
import matplotlib.pyplot as plt

data = np.fromfile(r"C:\Users\sacha\AppData\Local\Programs\Microsoft VS Code\elec.bin", dtype=np.int16) 


# FRAME_SIZE = 256
FRAME_SIZE = 20000 #affiche complet du signal
HOP_SIZE = 89

frame0 = data[0:FRAME_SIZE]
# frame1 = data[FRAME_SIZE:2*FRAME_SIZE]
# frame2 = data[2*FRAME_SIZE:3*FRAME_SIZE] #création des tableaux qui vont contenir les frames

ymin, ymax = -32000, 32000  # on fixe l'échelle sinon on ne voit pas bien les correspondances

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1) #affichage des 3 graphes les uns sous les autres (3 lignes 1 colonnes) (fig -> fenetre globale séparée en 3 sous parties)
fig, ax1 = plt.subplots(1, 1) 

ax1.plot(frame0) #affichage frame0
# ax1.axvline(x=FRAME_SIZE-HOP_SIZE, color='r', linestyle='--', label='début overlap')#on affiche une ligne sur l'overlap + sa légende
ax1.set_title("Frame 0") #titre
ax1.set_ylim(ymin, ymax) #mise à l'échelle
ax1.legend() #affichage légende

# ax2.plot(frame1)
# ax2.axvline(x=HOP_SIZE, color='r', linestyle='--', label='fin overlap')
# ax2.axvline(x=FRAME_SIZE-HOP_SIZE, color='g', linestyle='--', label='début overlap suivant')
# ax2.set_title("Frame 1")
# ax2.set_ylim(ymin, ymax)
# ax2.legend()

# ax3.plot(frame2)
# ax3.axvline(x=HOP_SIZE, color='g', linestyle='--', label='fin overlap')
# ax3.set_title("Frame 2")
# ax3.set_ylim(ymin, ymax)
# ax3.legend()

plt.tight_layout() #pour un affichage propre
plt.show() #affichage 