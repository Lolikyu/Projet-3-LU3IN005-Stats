import numpy as np
import math
import matplotlib.pyplot as plt
import random as rd

def rec_extract_values(iterable, dico):
    """
    Paramètres:
    iterable : any
        Iterable (ou non) dont on veut extraire les valeurs dans un dictionnaire
    dico : dict
        Dictionnaire qui stockera les valeurs extraites
        
    Retourne:
    None

    Description:
    Fonction permettant d'extraire toutes les valeurs de iterable en les stockant
    dans dico, et qui compte toutes les fois où chaque valeur est apparue.
    (Dans cette fonction les strings sont considérés comme non-iterable, donc
    seront compté comme une valeur quelconque)
    
    Exemple:
    dico : {}
    rec_extract_values([1, 3, [3, 7]])
    dico : {1:1, 3:2, 7:1}
    """
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
    """
    Paramètres:
    iterable : any iterable type of (int or float)
        Iterable dont on veut sommer les valeurs
        
    Retourne:
    somme : int or float
        Somme des valeurs de iterable
    
    Description:
    Fonction permettant de retourner la somme de toutes les valeurs de iterable.
    
    Exemple:
    rec_sum_values([1, 3, [3, 7.5]]) : 14.5
    """
    if type(iterable) in [type([]), type(np.matrix([])), type(np.array([]))]:
        somme = 0
        for l in iterable:
            somme += rec_sum_values(l)
        return somme
    else:
        return iterable
    
def dict_zero(dico):
    """
    Paramètres:
    dico : dict
        Dictionnaire dont on veut mettre toutes ses valeurs à 0
        
    Retourne:
    None
    
    Description:
    Fonction permettant pour un dictionnaire de conserver ses clés tout en fixant leur valeur à 0.
    
    Exemple:
    dico : {1:1, 3:2, 7:1}
    dict_zero(dico)
    dico : {1:0, 3:0, 7:0}
    """
    for key in dico:
        dico[key] = 0
        

matrice_transition = np.array([[2/3, 1/3,   0], 
                               [  0, 5/6, 1/6], 
                               [  0,   0,   1]])


data = np.loadtxt("data_exo_2022.txt")


def analyse_seq(sequence, opt_card={}):
    sequence = sequence.astype(int)
    
    if opt_card == {}:
        card = {elem : 0 for elem in set(sequence)}
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

print("Matrice de transition de l'exemple:\n")
print(matrice_proba_transition_liste([np.array([0., 0., 0., 1., 1., 1., 1., 1., 1., 2.])]))
#print("\nMatrice de transition des 5000 individus:\n")
#print(matrice_proba_transition_liste(data))



matrice_transition_modele1 = np.array([[0.92, 0.08,    0], 
                                       [   0, 0.93, 0.07], 
                                       [   0,    0,    1]])

def est_stochastique(matrice):
    epsilon = 0.00001
    for ligne in matrice:
        if abs(rec_sum_values(ligne) - 1) >= epsilon:
            return False
    return True

print("La matrice de transition de la partie I) est-elle stochastique ? :", est_stochastique(matrice_transition))

print("La matrice de transition de la partie II) est-elle stochastique ? :", est_stochastique(matrice_transition))


pi_0 = [0.90,
        0.10,
           0]

pi_1 = [0.90*0.92 + 0*0      ,
        0.90*0.08 + 0.10*0.93,
        0.10*0.07 + 0*1      ]

print("pi_1 =", pi_1)


pi_2 = [0.828*0.92 + 0.007*0   ,
        0.828*0.08 + 0.165*0.93,
        0.165*0.07 + 0.007*1   ]

print("pi_2 =", pi_2)


def liste_pi(pi_0, A, n):
    l_pi = [pi_0]
    for i in range(n):
        pi_i = []
        for j in range(len(pi_0)):
            pi_i.append(l_pi[-1][j] * A[j][j] + l_pi[-1][j-1] * A[j-1][j])
        l_pi.append(pi_i)
    return l_pi

#print("Liste des pi_i pour i entre 0 et 200 :")
#print(np.array(liste_pi(pi_0, matrice_transition_modele1, 200)))

liste_pi_i = np.array(liste_pi(pi_0, matrice_transition_modele1, 200))

def affiche_graphe_unique(liste_ordonnes, categorie, x_scale, is_ponctuel):
    """
    Paramètres:
    func : function
        Fonction dont on souhaite afficher le graphe
    s_scale : str
        Mot-clé "linear" ou "log" permettant de définir l'échelle de l'axe des abscisses
    is_ponctuel : bool
        Booléen qui décide de si oui ou non le graphe est ponctuel (si non, il est tracé via une courbe)    
        
    Retourne:
    : None

    Hypothèses:
        On suppose que temps_exec_file(func) a déjà été exécuté auparavant

    Description:
    Permet d'afficher le graphe du temps d'exécution de func, paramétré par x_scale et is_ponctuel
    """
    #on crée les abscisses
    abscisses = [i for i in range(200 +1)]
    #on crée les ordonnées
    ordonnees = liste_ordonnes
    
    #si on a l'option is_ponctuel à True
    if is_ponctuel:
        #alors on plot le graphe de manière ponctuelle, en prenant en compte du nombre de points
        plt.plot(abscisses, ordonnees, "x")
    else:
        #sinon, on plot le graphe comme une courbe, en prenant en compte du nombre de points
        plt.plot(abscisses, ordonnees)
        
    #on nomme l'axe des abscisses
    plt.xlabel("Temps t")
    #on nomme l'axe des ordonnées
    if categorie == "S":
        plt.ylabel("Proportion des personnes saines")
    elif categorie == "I":
        plt.ylabel("Proportion des personnes infectées")
    elif categorie == "R":
        plt.ylabel("Proportion des personnes guéries")
    #on paramètre l'échelle des abscisses suivant l'option x_scale
    plt.xscale(x_scale)
    #on paramètre l'échelle des ordonnées comme linéaire
    plt.yscale("linear")
    #on donne un titre au graphique
    if categorie == "S":
        plt.title("Graphique de la proportion de personnes saines en fonction du temps")
    elif categorie == "I":
        plt.title("Graphique de la proportion de personnes infectées en fonction du temps")
    elif categorie == "R":
        plt.title("Graphique de la proportion de personnes guéries en fonction du temps")
   
    #on affiche le graphe
    plt.show()


#affiche_graphe_unique(liste_sains, "S", "linear", False)
#affiche_graphe_unique(liste_infectes, "I", "linear", False)
#affiche_graphe_unique(liste_gueris, "R", "linear", False)


def affiche_graphe_multi(liste_pi):
    #on crée les ordonnées
    liste_sains = []
    liste_infectes = []
    liste_gueris = []
    for pi in liste_pi_i:
        liste_sains.append(pi[0])
        liste_infectes.append(pi[1])
        liste_gueris.append(pi[2])
    
    #on crée les abscisses
    abscisses = [i for i in range(len(liste_pi))]
    
    #on plot le graphe comme une courbe
    plt.plot(abscisses, liste_sains)
    plt.plot(abscisses, liste_infectes)
    plt.plot(abscisses, liste_gueris)
        
    #on nomme l'axe des abscisses
    plt.xlabel("Temps t")
    #on nomme l'axe des ordonnées
    plt.ylabel("Proportion")
    
    #on paramètre l'échelle des abscisses suivant l'option x_scale
    plt.xscale("linear")
    #on paramètre l'échelle des ordonnées comme linéaire
    plt.yscale("linear")
    
    #on donne un titre au graphique
    plt.title("Graphique des proportions de population en fonction du temps")
       
    #on affiche le graphe
    plt.show()

#affiche_graphe_multi(liste_pi_i)

mat_transition_modele2 = np.array([[0.92, 0.08,    0], 
                                   [   0, 0.93, 0.07], 
                                   [0.02,    0, 0.98]])