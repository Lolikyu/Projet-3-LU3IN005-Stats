import numpy as np
import math
import matplotlib.pyplot as plt
import random

matrice_transition = np.array([[2/3, 1/3,   0], 
                               [  0, 5/6, 1/6], 
                               [  0,   0,   1]])

seq_test = np.array([0., 0., 0., 1., 1., 1., 1., 1., 1., 2.])

data = np.loadtxt("data_exo_2022.txt")

def rec_extract_values(iterable, dico):
    if type(iterable) in [type([]), type(np.matrix([])), type(np.array([]))]:
        for l in iterable:
            rec_extract_values(l, dico)
        return dico
    else:
        if iterable not in dico:
            dico[iterable] = 1
        else:
            dico[iterable] += 1
            
def rec_sum_values(iterable):
    if type(iterable) in [type([]), type(np.matrix([])), type(np.array([]))]:
        somme = 0
        for l in iterable:
            somme += rec_sum_values(l)
        return somme
    else:
        return iterable

def analyse_seq(sequence, opt_card={}):
    sequence = sequence.astype(int)
    
    if opt_card == {}:
        card = {float(elem) : 0 for elem in set(sequence)}
        nb_etats = len(card)
        mat_transition_etats = np.zeros((nb_etats, nb_etats))
        
        #on détermine la fréquence
        for etat in range(1, len(sequence)):
            card[sequence[etat-1]] += 1
            mat_transition_etats[sequence[etat-1]][sequence[etat]] += 1
        card[sequence[etat]] += 1
        mat_transition_etats[sequence[etat]][sequence[etat]] += 1

    else:
        card = opt_card
        nb_etats = len(card)
        mat_transition_etats = np.zeros((nb_etats, nb_etats))
        
        #on détermine la fréquence
        for etat in range(1, len(sequence)):
            mat_transition_etats[sequence[etat-1]][sequence[etat]] += 1
        mat_transition_etats[sequence[etat]][sequence[etat]] += 1
        
    return mat_transition_etats, card
    
def matrice_proba_transition(sequence, opt_card={}):
    mat_transition_etats, card = analyse_seq(sequence, opt_card)
    #on détermine les probabilités
    nb_etats = len(mat_transition_etats)
    for i in range(nb_etats):
        for j in range (nb_etats):
            mat_transition_etats[i][j] /= card[i]
    return mat_transition_etats
    
def dict_zero(dico):
    for key in dico:
        dico[key] = 0
    return dico

def matrice_proba_transition_liste(liste_sequence):
    card = {}
    rec_extract_values(liste_sequence, card)
    
    matrice_finale = np.zeros((len(card), len(card)))
    
    for seq in liste_sequence:
        dict_zero(card)
        rec_extract_values(seq, card)
        matrice_finale += matrice_proba_transition(seq, card)
        
    for i in range(len(matrice_finale)):
        for j in range(len(matrice_finale[0])):
            matrice_finale[i][j] /= (len(liste_sequence))
            
    return matrice_finale
        

#print(matrice_proba_transition(seq_test))

#________________________________________________________________

#1) La matrice de transition est la suivante :
"""
[" ",  "S",  "I",  "R"]
["S", 0.92, 0.08,    0]
["I",    0, 0.93, 0.07]
["R",    0,    0,    1]
"""
mat_transition_modele1 = np.array([[0.92, 0.08,    0], 
                                   [   0, 0.93, 0.07], 
                                   [   0,    0,    1]])

def est_stochastique(matrice):
    epsilon = 0.00001
    for ligne in matrice:
        if abs(rec_sum_values(ligne) - 1) >= epsilon:
            return False
    return True

#2) Proportions de la population :
"""
pi_0 = [0.90,
       0.10,
       0   ]

pi_1 = [0.90*0.92 + 0*0,
       0.90*0.08 + 0.10*0.93,
       0.10*0.07 + 0*1      ]
#   = [0.828, 0.165, 0.007]

pi_2 = [0.828*0.92 + 0.007*0,
       0.828*0.08 + 0.165*0.93,
       0.165*0.07 + 0.007*1      ]
#   = [0.76176, 0.21969, 0.01855]
"""

#3)

def liste_pi(pi_0, A, n):
    l_pi = [pi_0]
    for i in range(n):
        pi_i = []
        for j in range(len(pi_0)):
            pi_i.append(l_pi[-1][j] * A[j][j] + l_pi[-1][j-1] * A[j-1][j])
        l_pi.append(pi_i)
    return l_pi
# print(liste_pi(pi_0, mat_transition_modele1, 200))

#4)



#________________________________________________________________
#1) Ce processus peut bien être modelisé par une chaîne de Markov
#2) La matrice de transition est la suivante :
"""
[" ",  "S",  "I",  "R"]
["S", 0.92, 0.08,    0]
["I",    0, 0.93, 0.07]
["R", 0.02,    0, 0.98]
"""
mat_transition_modele2 = np.array([[0.92, 0.08,    0], 
                                   [   0, 0.93, 0.07], 
                                   [0.02,    0, 0.98]])

