#!/usr/bin/env python
# coding: utf-8

# In[0]:


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import itertools

# create total state
basis_list = [basis(2, i) for i in range(2)]
ket_list = [tensor(basis_0, basis_1, basis_2, basis_3)
            for basis_0 in basis_list
            for basis_1 in basis_list
            for basis_2 in basis_list
            for basis_3 in basis_list]

# create annihilation operator
def des(site_index):
    ops = [destroy(2) if i == site_index else sigmaz() if i < site_index else qeye(2) for i in range(4)]
    return tensor(ops)

des_op = [des(i) for i in range(4)]
n1 = des_op[0].dag() * des_op[0] + des_op[1].dag() * des_op[1]
n2 = des_op[2].dag() * des_op[2] + des_op[3].dag() * des_op[3]
ide = tensor(qeye(2),qeye(2),qeye(2),qeye(2))

# 參數設置
t = 0.1
U = 2.0  
V = 0.5 
e1 = 0
hopping = -t * (des_op[0].dag() * des_op[2] + des_op[2].dag() * des_op[0] +
                     des_op[1].dag() * des_op[3] + des_op[3].dag() * des_op[1])
interaction = U * (n1 * (n1 - ide) / 2 + n2 * (n2 - ide) / 2)
onsite_potential = V * n1 * n2

def detuning(e1, d):
    return -((e1 - d) * n1 + (e1 + d) * n2)

H = hopping + interaction + onsite_potential

# detuning 範圍
d_range = np.linspace(-3, 3, 1000)

energies_vs_detuning = []

# 生成所有具有兩個位元為 1 的二進位數字並排序
binary_combinations = sorted(list(itertools.combinations(range(4), 2)))

# 轉換成對應的整數
selected_ket_indices = [int(''.join('1' if i in combination else '0' for i in range(4)), 2) for combination in binary_combinations]
selected_ket_indices = selected_ket_indices[::-1]
print(selected_ket_indices)

# 對每個 detuning 值，找出我們選的 Hamiltonian
for d in d_range:
    H_detuned = H + detuning(e1, d)
    subspace_H_detuned = H_detuned.extract_states(selected_ket_indices)
    eigenvalues_detuned = subspace_H_detuned.eigenenergies()
    energies_vs_detuning.append(eigenvalues_detuned)

energies_vs_detuning = np.array(energies_vs_detuning)

# 繪製能量 vs detuning 圖
for n in range(len(selected_ket_indices)):
    plt.plot(d_range, energies_vs_detuning[:, n], label=selected_ket_indices[n])
plt.xlabel("Detuning")
plt.ylabel("Energy")
plt.title("Energy vs Detuning")
plt.legend()
plt.show()





