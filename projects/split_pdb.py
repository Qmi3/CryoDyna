import os
import sys
pca_pdb = sys.argv[1]
pca_dir = pca_pdb.replace('.pdb', '')
os.makedirs(pca_dir, exist_ok=True)
with open(pca_pdb, 'r') as f:
    lines = f.readlines()
idx = 1

for line in lines:
    if line.startswith('MODEL'):
        f = open(f'{pca_dir}/{idx}.pdb', 'w')
        f.write(line)
    elif line.startswith('ENDMDL'):
        f.write(line)
        f.close()
        idx += 1
    elif line.startswith('CRYST'):
        f.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1 \n")
    else:
        f.write(line)    
