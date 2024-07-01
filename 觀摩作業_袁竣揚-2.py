# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:29:29 2024

@author: Asher
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
#%%
#create total state
# ket = [basis(2, i) for i in range(2)]
basis_list = [basis(2, i) for i in range(2)]
ket_list = []
for basis_0 in basis_list:
    for basis_1 in basis_list:
        for basis_2 in basis_list:
            for basis_3 in basis_list:
                ket_list.append(tensor(basis_0, basis_1, basis_2, basis_3))
print(ket_list)
#%%
#create anihilation operator
def des(site_index):
    ops = [destroy(2) if i == site_index else sigmaz() if i < site_index else qeye(2) for i in range(4)]
    return tensor(ops)

# 建立每個 site 的 annihilation operator 並 tensor 成矩陣
des_op = [des(i) for i in range(4)]
n1 = des_op[0].dag() * des_op[0] + des_op[1].dag() * des_op[1]
n2 = des_op[2].dag() * des_op[2] + des_op[3].dag() * des_op[3]
ide = tensor(qeye(2),qeye(2),qeye(2),qeye(2))
#%%
t = 0.1
U = 2.0  
V = 0.5 
e1 = 0
hopping_term = -t * ((des_op[0].dag()* des_op[2]) + (des_op[2].dag()* des_op[0]) +
                 (des_op[1].dag()* des_op[3]) + (des_op[3].dag()* des_op[1]))

interaction_term = U * ((n1 * (n1 - ide) / 2 + n2 * (n2 - ide) / 2))

onsite_potential = V*n1*n2

def detuning(e1, d):
    return -((e1-d)*n1 + (e1+d)*n2)
H = hopping_term + interaction_term + onsite_potential


#%%
# detuning 範圍
d_range = np.linspace(-3, 3, 500)

energies_vs_detuning = []
#對應到二進制中的state, ex:1100為12
selected_ket_indices = [3, 5, 6, 9, 10, 12]
#  對每個detuning 值， 找出我們選的Hamiltonian，
for d in d_range:
    H_detuned = H + detuning(e1, d)
    subspace_H_detuned = H_detuned.extract_states(selected_ket_indices)
    
    eigenvalues_detuned = subspace_H_detuned.eigenenergies()
    
    energies_vs_detuning.append(eigenvalues_detuned)

energies_vs_detuning = np.array(energies_vs_detuning)

for n in range(len(selected_ket_indices)):
    plt.plot(d_range, energies_vs_detuning[:, n], label= selected_ket_indices[n])
plt.xlabel("Detuning")
plt.ylabel("Energy")
plt.title("Energy vs Detuning")
plt.legend()
plt.show()
