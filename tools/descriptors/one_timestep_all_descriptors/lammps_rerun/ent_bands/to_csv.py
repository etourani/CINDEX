import numpy as np 
import re 
import pandas as pd
import matplotlib.pyplot as plt

atoms_file = './dump_rerunS_noavg_t2'
with open(atoms_file) as f:
  lines = f.readlines()

# Find N the number of atoms 
N= int(lines[next(l[0] for l in enumerate(lines) if re.match('^ITEM: NUMBER OF ATOMS', l[1]))+1])
print("Find N, the number of atoms: ", N)

# Read starting timestep (t0)
l0= next(l[0] for l in enumerate(lines) if re.match('ITEM: TIMESTEP', l[1]))+1
t0= int(lines[l0])
print("starting timestep (t0): ", t0)

# Extracting the box dimensions
b0 = next(l[0] for l in enumerate(lines) if re.match('ITEM: BOX BOUNDS', l[1]))+1
print('b0: ', b0)

# Find number of lines per timestep (l1-l0)
is_second_match= False
for l in enumerate(lines):
    if re.match('ITEM: TIMESTEP', l[1]):
        if is_second_match:
            l1= l[0]+1
            t1= int(lines[l1])
            dt= t1 - t0  # timestep
            break
        else:
            is_second_match= True
print("timestep, dt: ", dt)

# Read last time (te)
le= len(lines) - (l1 - l0) + l0
te= int(lines[le])
print("last time, te: ", te)
Ns= int((te - t0)/dt + 1)
print("Ns: ", Ns)
print(Ns*(l1-l0) == len(lines)) # sanity check

# to round the precision of xlo, xhi, ylo, etc
def round_to_precision(value, precision=3):
    return round(value, precision)

# Write csv file
a0= next(l[0] for l in enumerate(lines) if re.match('ITEM: ATOMS', l[1]))+1
with open('dumpS_noavg_t2.csv', 'w') as fl:
    tmp= fl.write('step,t,atom_id,mol_id,x,y,z,S,xlo,xhi,ylo,yhi,zlo,zhi\n')
    for step in range(Ns):
        t= (t0 + step*dt) #/ 1000  
        # Getting the box bounds
        box_idx = step*(l1-l0) + b0

        xlo, xhi = map(lambda x: round_to_precision(float(x)), lines[box_idx].split()[:2])
        ylo, yhi = map(lambda x: round_to_precision(float(x)), lines[box_idx+1].split()[:2])
        zlo, zhi = map(lambda x: round_to_precision(float(x)), lines[box_idx+2].split()[:2])

        for i in range(N):
            idx= step*(l1-l0) + a0 + i
            atom_id= int(lines[idx].split(' ')[0])
            mol_id= float(lines[idx].split(' ')[1])
            atom_x= float(lines[idx].split(' ')[2])
            atom_y= float(lines[idx].split(' ')[3])
            atom_z= float(lines[idx].split(' ')[4]) 
            #atom_entr= float(lines[idx].split(' ')[8])
            atom_entr= round_to_precision(float(lines[idx].split(' ')[5]))



            if i == 0:
                tmp= fl.write(str(step)+','+str(t)+','+str(atom_id)+','+
                          str(mol_id)+','+str(atom_x)+','+str(atom_y)+','+
                          str(atom_z)+','+str(atom_entr)+',' +str(xlo)+','+str(xhi)+','+str(ylo)+','+str(yhi)+','+str(zlo)+','+str(zhi)+'\n')
            else:
                tmp = fl.write(str(step)+','+str(t)+','+str(atom_id)+','+ 
                          str(mol_id)+','+str(atom_x)+','+str(atom_y)+','+
                          str(atom_z)+','+str(atom_entr)+','+str(0)+','+str(0)+','+str(0)+','+str(0)+','+str(0)+','+str(0)+'\n')

