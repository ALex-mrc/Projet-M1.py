# -*- coding: utf-8 -*-

import json
import requests
from xml.dom import minidom
import numpy as np
import math
from numpy import array
import cv2
import json


# -- PARTIE 1 : LECTURE DES DONNEES (URL) --

def getNbFeatureEmprise(url, bbox):
    nb = 0
    proxyDict = {
    'http': 'http://10.0.4.2:3128',
    'https': 'http://10.0.4.2:3128',
    }

    url_hits = url
    url_hits += "BBOX=" + bbox
    url_hits += "&resulttype=hits"

    response = requests.get(url_hits) #response = requests.get(url_hits, proxies=proxyDict)
    http_status = response.status_code
    
    if http_status == 200:
        data = response.content
        dom = minidom.parseString(data)
        result = dom.getElementsByTagName("wfs:FeatureCollection")
        nb = int(result[0].attributes["numberMatched"].value)
    
    return nb
    
def get_street_feature_from_url(url, bbox):

    nb_per_page = 1000
    proxyDict = {
    'http': 'http://10.0.4.2:3128',
    'https': 'http://10.0.4.2:3128',
    }
    nbelem = getNbFeatureEmprise(url)
    nbiter = int(nbelem / nb_per_page) + 1
    offset = 0
    L_entite = []

    
    for j in range(nbiter):
        url_feat = url
        url_feat += "BBOX=" + bbox
        url_feat += "&count=" + str(nb_per_page)
        url_feat += "&startIndex=" + str(offset)
        url_feat += "&RESULTTYPE=results"

        response = requests.get(url_feat, proxies=proxyDict)
        http_status = response.status_code
    
        if http_status == 200:
            data = json.loads(response.text)
            features = data["features"]

            for feature in features:
               
                coords = feature["geometry"]["coordinates"]
                if feature["geometry"]["type"] == "LineString":
                    L_entite.append(coords)
    
    return (L_entite)

def get_bat_feature_from_url(url, bbox):

    nb_per_page = 1000
    proxyDict = {
    'http': 'http://10.0.4.2:3128',
    'https': 'http://10.0.4.2:3128',
    }
    nbelem = getNbFeatureEmprise(url, bbox)
    nbiter = int(nbelem / nb_per_page) + 1
    offset = 0
    L_entite = []

    
    for j in range(nbiter):
        #print("PAGE " + str(j + 1) + "/" + str(nbiter))
        url_feat = url
        url_feat += "BBOX=" + bbox
        url_feat += "&count=" + str(nb_per_page)
        url_feat += "&startIndex=" + str(offset)
        url_feat += "&RESULTTYPE=results"

        response = requests.get(url_feat, proxies=proxyDict)
        http_status = response.status_code
    
        if http_status == 200:
            data = json.loads(response.text) 
            features = data["features"]

            for feature in features:
               
                coords = feature["geometry"]["coordinates"][0][0] #extraction du premier polygone de chaque premier multi polygone
                if feature["geometry"]["type"] == "MultiPolygon": #sélection de la géométrie sous forme de multipolygone

                    L_entite.append(coords)
                        
    return (L_entite)

def get_feature_from_json(json_file):
    """ recuperation des multi polygones ou des polylignes d'un json

    Args:
        json_file (_string_): chemin du fichier json

    Return : L_entite (_list_): liste des coordonnes des entites du json
    """
    
    L_entite = []

    with open(json_file, 'r') as data1:
        data2 = json.load(data1)
        
        features = data2["features"]

        for feature in features:
            coords = feature["geometry"]["coordinates"][0][0]
            if feature["geometry"]["type"] == "MultiPolygon" or feature["geometry"]["type"] == "LineString":
                L_entite.append(coords)

    return(L_entite)


# -- PARTIE 2 : SELECTION DES BATIMENTS RETENUS POUR LE CALCUL --

def barycentre(polygone):
    """ calcul du barycentre d'un polygone 3D projete dans le plan (pas de dimension z)

    Args:
        polygone (_list_): liste des points qui définiseent le polygone (liste de liste)

    Return : barycentre (_list_): coordonnées x et y du barycentre
    """
    
    liste_x = []
    liste_y = []

    for sommet in polygone:
        liste_x.append(sommet[0]) #recuperation des x de tous les sommets
        liste_y.append(sommet[1]) #recuperation des y de tous les sommets
    
    avg_x = sum(liste_x) / len(liste_x) #moyenne des x
    avg_y = sum(liste_y) / len(liste_y) #moyenne des y
    barycentre = [avg_x, avg_y]

    return barycentre

def calc_equa_droite(p1,p2):
    """ calcul de l'equation de la droite passant par p1 et p2 

    Args:
        p1 (_tuple_): premier point de definition du segment
        p2 (_tuple_): deuxieme point de definition du segment

    Return : (a, b) (_tuple_): coeff de la droite et ordonnee a l'origine
    """

    try : 
        a = (p1[1] - p2[1]) / (p1[0] - p2[0]) #calcul du coeff
        b = p1[1] - a * p1[0] #calcul de ordonnee a l'origine
    except ZeroDivisionError:
        (a,b) = (-1,-1)
    return (a,b)

def dist_point_seg(P, D):
    """calcul de la distance d'un point à une droite

    Args:
        P (_list_): liste des deux coordonees du point
        D (_tuple_): equation d'une droite, sous la forme (coeff, ordonnee_origine)

    Return : dist_pt_seg (_float_): distance en metres entre P et D
    """

    dist_pt_seg = (np.abs(D[0]*P[0] - P[1]+D[1]))/(np.sqrt(1+D[0]**2))

    return dist_pt_seg

def barycentre_liste_poly(liste_poly):
    """calcul de tous les barycentres d'une liste de polygone

    Args:
        liste_poly (_list_): liste des polygones

    Return : bary_list_poly: liste contenant chaque polygone, associe a la valeur de son barycentre
    """

    element_liste =[]
    bary_liste_poly = []


    for i in range(len(liste_poly)): #pour chaque polygone
        element_liste.append(liste_poly[i]) #ajout a la sous liste du polygone courant
        element_liste.append(barycentre(liste_poly[i])) #ajout a la sous liste du barycentre du polygone courant
        bary_liste_poly.append(element_liste) #ajout de la sous liste dans le conteneur

        element_liste = [] #reinitialisation de la sous-liste

    return(bary_liste_poly)

def polyligne_to_segment(polyligne):
    """decoupage d'une polyligne en segments

    Args:
        polyligne (_point_): liste de points constituant la polyligne

    Return : liste_segm (_list_): liste des segments de la polyligne
    """
    
    liste_segm = []
    liste_p1 =[]

    for i in range (len(polyligne)-1): 
        liste_p1.append(polyligne[i])  #ajout du point A du segment courant
        liste_p1.append(polyligne[i+1]) #ajout du point B du segment courant
        liste_segm.append(liste_p1) #ajoute le segment au conteneur
        liste_p1 = [] #reinitialisation de la liste contenant le segment
    return(liste_segm)

def distance_batiment_polyligne(polyligne, bary_liste_polygone):
    """pour chaque element (constitué d'un polygone et de son barycentre), renvoie un conteneur ou le polygone est associe a sa
    distance au segment le plus proche de la polyligne choisie

    Args:
        polyligne (_list_): polyligne d'etude
        bary_liste_polygone (_list_): conteneur ou chaque element associe un polygone et son barycentre

    Return : cont_distance_polygone_polyligne (_list_): liste des polygones et de leur distance minimale à la polyligne
    """ 

    distance_min = 100000 #initialisation de la distance min (en m) a une valeur qui depasse la taille de la BBOX
    liste_temp = []
    cont_distance_polygone_polyligne =[]
    liste_segment = polyligne_to_segment(polyligne)

    for k in range(len(bary_liste_polygone)): #pour chaque element (constitué d'un polygone et de son barycentre)
        temp_barycentre_x = bary_liste_polygone[k][1][0] #recuperation du x du barycentre
        temp_barycentre_y = bary_liste_polygone[k][1][1] #recuperation du y du barycentre
        temp_barycentre= (temp_barycentre_x,temp_barycentre_y) #tuple avec les coordonnees du barycentre du polygone en cours 

        for i in range(len(liste_segment)): #pour chaque element du conteneur recherche du segment de la polyligne le plus pres du barycentre de l'entite k
            XA = liste_segment[i][0][0] # recuperation du XA du segment i  
            XB = liste_segment[i][1][0] #        -        XB            -
            YA = liste_segment[i][0][1] #        -        YA            -
            YB = liste_segment[i][1][1] #        -        YB            -
            A = (XA,YA) # coordonnees du point A du segment i
            B = (XB,YB) #           -          B           -

            a_segment,b_segment= calc_equa_droite(A,B) #calcul de l'equation du segment i
            equation_seg = (a_segment,b_segment)
                
            distance = dist_point_seg(temp_barycentre, equation_seg) #calcul de la distance entre ce segment et le barycentre courant

            if distance < distance_min: #si cette distance est inferieure a la distance min temporaire   
                distance_min = distance #recuperation de la disatnce min

        polygone_temp = bary_liste_polygone[k][0] #recuperation du polygone correspondant au barycentre courant
        liste_temp.append(polygone_temp) # dans une liste temporaire : ajout du polygone...
        liste_temp.append(distance_min) #...et de sa distance au segment le plus proche...
        cont_distance_polygone_polyligne.append(liste_temp) #l'ensemble est ajoute au conteneur associant chaque polygone au segment le plus proche
        
        distance_min = 100000 #reinitialisation de la distance min pour le barycentre suivant
        liste_temp = [] #reinitialisation de la liste temporaire
                          
    return(cont_distance_polygone_polyligne)

def buffer(cont_distance_polygone_polyligne, dist_max):
    """recuperation, depuis le conteneur des distances, des batiments situes a une distance max de <dist_max> autour de la polyligne 

    Args:
        cont_distance_polygone_polyligne (_list_): conteneur ou chaque batiment a ete associe a sa distance minimale a la polyligne
        dist_max (_list_): distance maximale toleree pour associer le batiment a la polyligne

    Return : batiments_dans_buffer (_list_): liste des batiments dont le barycentre est contenu dans le buffer
    """

    batiments_dans_buffer = [] #liste qui contiendra les batiments contenus dans le buffer

    for i in range(len(cont_distance_polygone_polyligne)): #pour chaque batiment
        distance = cont_distance_polygone_polyligne[i][1] #recuperation de sa distance minimale a la polyligne

        if distance <= dist_max: #si cette distance est inferieur ou egale a la tolerence
            batiments_dans_buffer.append(cont_distance_polygone_polyligne[i][0]) #ajoute le polygone a la liste

    return batiments_dans_buffer

def suppression_polygone(liste_polygone):
    """a partir d'une liste de polygones, ne conserve que les polygones dont les sommets ont un z enregistre (z != -1000) 

    Args:
        liste_polygone (_list_): liste des polygones a nettoyer

    Return : liste_propre, la liste ou chaque polygone a des sommets dont le z est different de -1000
    """

    liste_propre = []
    for i in range(len(liste_polygone)): #pour chaque polygone de la liste...
        z = liste_polygone[i][0][2] #recupere le z du premier sommet
        if z != -1000: #si le z a une valeur differente de -1000
            liste_propre.append(liste_polygone[i]) #ajoute le polygone correspondant a la liste propre
    
    return liste_propre

def liste_bat_candidats(liste_polygone, route, dist_max):
    """a partir d'une liste de polygone, determine lesquels sont candidats a la silhouette (dans le buffer et avec un z != -1000)

    Args:
        liste_polygone (_list_): liste des batiments de depart
        route (_polyligne_): liste des points qui composent la route etudiee
        dist_max (_float_): distance entre la route et le batiment en dessous de laquelle le batiment sera pris en compte dans le calcul de silhouette (=buffer)

    Return : batiments_candidats (_list_): liste de polygones situes dans le buffer defini autour de la route
    """

    liste_polygone_propre = suppression_polygone(liste_polygone) #suppression des polygones sans z
    liste_barycentre = barycentre_liste_poly(liste_polygone_propre) #calcul des barycentres des batiments
    dist_batiment_route = distance_batiment_polyligne(route, liste_barycentre) #calcul des distances a la route
    batiments_candidats = buffer(dist_batiment_route, dist_max) #selection des batiments dans le buffer

    return batiments_candidats


# -- PARTIE 3 : CALCUL DES RAYONS --

def def_base_rayons(segment, precision_min):
    """a partir d'un segment (en 3D), definit les coordonnees (en 3D) des points de ce segment qui seront à l'origine des rayons.

    Args:
        segment (_list_): contient les coordonnees des extremites du segment (en x, y et z)
        precision_min (_float_): precision minimale (en metres). Sera equivalent a la resolution de la silhouette de sortie sur l'axe des x.

    Return : bases_rayon, la liste des points (coord x, y et z) du segment a partir desquel les rayons seront projetés.
    """
    bases_rayons = []
    bases_rayons.append(segment[0]) #ajout du premier point, l'extremite du segment

    XA = segment[0][0]
    YA = segment[0][1]
    ZA = segment[0][2]
    XB = segment[1][0]
    YB = segment[1][1]
    ZB = segment[1][2]
    distance_a_b = math.sqrt(((XA-XB)**2)+((YA-YB)**2)+((ZA-ZB)**2)) #calcul de la longueur du segment

    pas = round((distance_a_b/precision_min)+1) #definition du nombre de pas en fonction de la precision souhaitee
    n = int(pas) #conversion en int pour la boucle for

    pas_x = (XB-XA)/(n+1) #pas x, y et z pour un decoupage du segment en n+1 portions, de longueur inferieure à la precision
    pas_y = (YB-YA)/(n+1)
    pas_z = (ZB-ZA)/(n+1)

    for i in range(n): #creation de n points intermédiaires
         point_temp_x = XA + (i+1)*pas_x
         point_temp_y = YA + (i+1)*pas_y
         point_temp_z = ZA + (i+1)*pas_z
         point_temp = [point_temp_x, point_temp_y, point_temp_z]
         bases_rayons.append(point_temp)

    return bases_rayons

def def_origines_rayons(points_base_rayons, precision_min, hauteur_max):
    """A partir d'une liste de points, calcule pour chaque point toutes ses projections sur l'axe des z avec une resolution
    de <precision_min>, jusqu'a une hauteur de <hauteur_max>

    Args:
        points_base_rayons (_list_): liste des points sources (coord en x, y et z). Ils appartiennent à la route.
        precision_min (_float_): precision minimale (en metres). Sera equivalent a la resolution de la silhouette de sortie sur l'axe des z.
        hauteur_max (_floay_): hauteur maximale jusqu'a laquelle les points d'origines seront calcules. Elle est exprimee en hauteur relative par rapport a la route.

    Return : origines_rayons, la liste des points (coord x, y et z) a partir desquel les rayons seront projetés vers les batiments.
    """

    origines_rayons = [] #toutes les origines
    nb_colonnes = len(points_base_rayons)

    n_pas_z = round((hauteur_max/precision_min)+1)  #definition du nombre de pas en fonction de la precision souhaitee
    dist_pas_z = hauteur_max/(n_pas_z+1) #calcul de la distance d'un pas sur l'axe des z. Cette distance est inferieure a la precision min

    for i in range(len(points_base_rayons)): #pour chaque point de base de la route
        origines_rayons.append(points_base_rayons[i]) #ajout du point au niveau de la route
        point_temp_x = points_base_rayons[i][0] #meme coord x que le point de base
        point_temp_y = points_base_rayons[i][1] #meme coord y que le point de base

        for j in range(n_pas_z): #creation de <n_pas_z> projections du point courant
            
            point_temp_z = points_base_rayons[i][2] + (j+1)*dist_pas_z #coord z du point de base + n fois la distance du pas
            point_temp = [point_temp_x, point_temp_y, point_temp_z]

            origines_rayons.append(point_temp)

        point_temp_z = points_base_rayons[i][2] + hauteur_max #coord z du point de base + hauteur max
        point_temp = [point_temp_x, point_temp_y, point_temp_z] 

        origines_rayons.append(point_temp) #ajout du dernier point (le plus haut)

    return origines_rayons, nb_colonnes

def calc_origines_rayons_polyligne(polyligne, precision_min, hauteur_max):
    """A partir d'une polyligne, retourne un conteneur ou chaque segment de la polyligne est associe a la liste des origines de rayons correspondants 

    Args:
        polyligne (_list_): polyligne (liste de points en coord x, y et z) a partir de laquelle sont calculees les origines
        precision_min (_float_): precision minimale (en metres). Sera equivalent a la resolution de la silhouette de sortie sur l'axe des z et des x.
        hauteur_max (_floay_): hauteur maximale jusqu'a laquelle les points d'origines seront calcules. Elle est exprimee en hauteur relative par rapport a la route.

    Return : conteneur_origines_par_segment, les segments de la polyligne, associes a la liste des points (coord x, y et z) a partir desquel les rayons seront projetés vers les batiments.
    """

    conteneur_origines_par_segment = []
    compt_colonnes = 0
    elem_conteneur = [] #une association entre un segment et toutes les origines associees
    segments = polyligne_to_segment(polyligne) #recuperation des segments de la polyligne

    for elem in segments: #pour chaque segment de la polyligne
        base_rayons = def_base_rayons(elem, precision_min) #calcul des origines au niveau de la route
        origines_rayons, nb_colonnes = def_origines_rayons(base_rayons, precision_min, hauteur_max) #ajout des origines sur l'axe z
        compt_colonnes = compt_colonnes + nb_colonnes #comptage du nombre de colonne dans la future image

        elem_conteneur.append(elem) #les coord des extremites du segment sont enregistrees
        elem_conteneur.append(origines_rayons) #puis les origines associees a ce segment
        conteneur_origines_par_segment.append(elem_conteneur) #le tout est ajoute dans le conteneur
        elem_conteneur = []

    return conteneur_origines_par_segment, compt_colonnes

def calc_equa_rayon(point_origine, segment):
    """Calcul de l'equation de la droite qui passe par le point origine et qui est perpendiculaire au segment, en deux dimensions. Cette droite correspond au rayon.

    Args:
        point_origine (_list_): x, y et z du point origine du rayon (le z ne sera pas pris en compte)
        segment (_list_): les deux point qui définissent le segment sur lequel est situe l'origine du rayon (le z ne sera pas pris en compte)

    Return : a2, b2 (_float_): le coef directeur et l'ordonnee a l'origine du rayon correspondant au point origine
    """

    a1, b1 = calc_equa_droite(segment[0], segment[1]) #calcul de l'equation du segment
    try :
        a2 = -1/a1 #calcul du coeff directeur du rayon
        b2 = point_origine[1] - (a2*point_origine[0]) #calcul de l'ordonee a l'origine du rayon
    except ZeroDivisionError:
        (a2,b2) = (-1000,-1000)

    return (a2, b2)


# -- PARTIE 4 : CALCUL DES INTERSECTIONS RAYONS/BATIMENTS --

def calc_equa_poly(polygone):
    """calcul de l'equation de chaque segment composant le polygone, neglige la dimension z

    Args:
        polygone (_list_): liste des sommets du polygone (seuls x et y seront pris en compte)

    Return : liste_equation, la liste des segments du polygone associes a leur "a" (coeff directeur) et leur "b" (ordonnee a l'origine)
    """

    segment = []
    liste_elem = []
    liste_equation = []

    for i in range(len(polygone)-1): #pour chaque sommet
        a, b = calc_equa_droite(polygone[i], polygone[i+1]) #calcul de a et b pour la droite passant par les sommets i et i+1
        segment.append(polygone[i])
        segment.append(polygone[i+1]) #creation du segment correspondant
        
        liste_elem.append(segment) #ajout du segment...
        liste_elem.append(a) #...et de son equation
        liste_elem.append(b)
        liste_equation.append(liste_elem) #ajout de l'element a la liste
        segment = []
        liste_elem = []
    
    return liste_equation

def calc_equa_list_poly(liste_polygone):
    """calcul de l'equation de chaque segment composants tous les polygones de la liste

    Args:
        liste_polygone (_list_): liste des polygones

    Return : liste_equation, la liste des segments composant les polygones, associes a leur "a" (coeff directeur) et leur "b" (ordonnee a l'origine)
    """

    liste_equation = []
    for i in range(len(liste_polygone)):
        equation = calc_equa_poly(liste_polygone[i])
        for j in range(len(equation)): #pour chaque segment+equation du polygone
            liste_equation.append(equation[j]) #ajout de cet element a la grande liste (enleve un niveau d'imbrication)

    return liste_equation

def point_dans_segment(point, segment):
    """verifie que le point appartient au segment dans un espace en 2 dimensions

    Args:
        point (_list_): x, y et z du point (le z ne sera pas pris en compte)
        segment (_list_): les deux point qui définissent le segment (le z ne sera pas pris en compte)

    Return : point_dedans (_bool_): retourne vrai si le point appartient au segment
    """
    point_dedans = False
    a, b = calc_equa_droite(segment[0], segment[1]) #calcul de l'equation du segment
    y_test = (a*point[0]) + b

    if y_test == point[1]: #si le point appartient à la droite
        x_min = min(segment[0][0], segment[1][0])
        x_max = max(segment[0][0], segment[1][0])

        if x_min <= point[0] and point[0] <= x_max: #si le point appartient au segment
            point_dedans = True

    return point_dedans

def distance_points(p1, p2):
    """calcul de la distance entre les points p1 et p2

    Args:
        p1 (_list_): x, y et z du point A (le z ne sera pas pris en compte)
        p2 (_list_): x, y et z du point B (le z ne sera pas pris en compte)

    Return : distance (_float_): distance en metres qui separe A et B
    """

    XA = p1[0]
    XB = p2[0]
    YA = p1[1]
    YB = p2[1]

    distance = np.sqrt(((XB-XA)**2)+((YB-YA)**2))

    return distance

def position_point_droite(point, segment):
    """determine si le point est situe a droite ou a gauche de la droite (AB) (c-a-d du vecteur AB)

    Args:
        point (_list_): x, y et z du point a determiner
        segment (_list_): points A et B du segment

    Return : position (_bool_): renvoie True si le point est a gauche
    """

    position = False

    XA = segment[0][0]
    YA = segment[0][1]
    XB = segment[1][0]
    YB = segment[1][1]
    XC = point[0]
    YC = point[1]

    d = ((XB-XA)*(YC-YA)) - ((YB-YA)*(XC-XA))

    if d > 0:
        position = True
    
    return position

def intersection_rayon_batiments(point_origine, segment_origine, liste_batiments):
    """pour chaque cote de la rue, renvoie l'intersection la plus proche entre le rayon issu de l'origine et la liste de batiments

    Args:
        point_origine (_list_): x, y et z du point origine du rayon
        segment_origine (_list_): les deux point qui définissent le segment sur lequel est place le point origine
        liste_batiments (_list_): liste des polygones associes au segment origine

    Return : intersection_gauche, intersection_droite (_list_): coordonnees x, y, z de l'intersection entre le rayon et le segment le plus proche
            Les intersections a gauche de la route sont dans la premiere liste, celles a droite sont dans la deuxieme
    """
    
    liste_intersection = []
    liste_intersection_gauche = []
    liste_intersection_droite = []
    liste_distances = []
    liste_equation_batiment = calc_equa_list_poly(liste_batiments) #calcul des equations de tous les segments qui appartiennent aux batiments
    a_rayon, b_rayon = calc_equa_rayon(point_origine, segment_origine) #calcul de l'equation du rayon passant par le point origine

    for i in range(len(liste_equation_batiment)): #pour chaque segment appartenant a un batiment
        a_bat = liste_equation_batiment[i][1] #recuperation du coeff directeur du segment
        b_bat = liste_equation_batiment[i][2] #recuperation de l'ordonne a l'origine du segment

        if a_bat != a_rayon: #si les deux droites sont secantes
            xI = (b_bat-b_rayon)/(a_rayon-a_bat) #calcul de l'intersection
            yI = (a_rayon*xI) + b_rayon
            zI = point_origine[2]

            appartient_segment = point_dans_segment([xI, yI, zI], liste_equation_batiment[i][0]) #verif que l'intersection est dans le segment en 2D

            if appartient_segment: #si l'intersection est sur le segment (en 2D)

                if zI <= liste_equation_batiment[i][0][0][2]: #si l'intersection est bien situee sur la facade
                    I = [xI, yI, zI]
                    liste_intersection.append(I) #ajout du point a la liste des intersections
        

    for k in range(len(liste_intersection)): #pour chaque intersection
        position_gauche = position_point_droite(liste_intersection[k], segment_origine) #calculer sa position relative au segment (=de quel cote de la rue)
        if position_gauche: #si l'intersection est a gauche
            liste_intersection_gauche.append(liste_intersection[k]) #elle est ajoutee dans la liste de gauche
        else:
            liste_intersection_droite.append(liste_intersection[k]) #sinon elle est ajoutee dans la liste de droite

    liste_intersection_gauche.append([0,0,0]) #dans les deux listes, ajout d'une valeur "temoin" qui fera office de valeur nulle en cas d'absence d'intersection
    liste_intersection_droite.append([0,0,0])

    for j in range(len(liste_intersection_gauche)): #pour chaque intersection trouvee a gauche de la route
        dist_temp = distance_points (point_origine, liste_intersection_gauche[j]) #calcul de sa distance au point origine
        liste_distances.append(dist_temp) #et ajout à une liste

    min_dist = min(liste_distances) #recuperation de la distance min
    index_min = liste_distances.index(min_dist) #recuperation de l'index de la distance min
    intersection_gauche = liste_intersection_gauche[index_min] #recuperation des coordonnees du point correspondant
    liste_distances = []

    for l in range(len(liste_intersection_droite)): #pour chaque intersection trouvee a droite de la route
        dist_temp = distance_points (point_origine, liste_intersection_droite[l]) #calcul de sa distance au point origine
        liste_distances.append(dist_temp) #et ajout à une liste

    min_dist = min(liste_distances) #recuperation de la distance min
    index_min = liste_distances.index(min_dist) #recuperation de l'index de la distance min
    intersection_droite = liste_intersection_droite[index_min] #recuperation des coordonnees du point correspondant
    liste_distances = []

    return intersection_gauche, intersection_droite

def liste_distance(batiments_candidats, route, precision_min, hauteur_max):
    """a partir d'une liste de polygones candidats et d'une route d'etude, determine les valeurs de toutes les distances entre origines et intersections

    Args:
        batiments_candidats (_list_): liste des batiments avec lesquels des intersections sont recherchees
        route (_polyligne_): liste des points qui composent la route etudiee
        precision_min (_float_): precision minimale (en metres). Sera equivalent a la resolution de la silhouette de sortie sur l'axe des z et des x.
        hauteur_max (_float_): hauteur maximale jusqu'a laquelle les points d'origines seront calcules. Elle est exprimee en hauteur relative par rapport a la route.

    Return : liste_pixels_gauche, liste_pixels_droite (_list_): deux listes, contenant respectivement la valeur du pixel gauche et du pixel droite 
    """
    
    origines_rayons, nb_colonnes = calc_origines_rayons_polyligne(route, precision_min, hauteur_max) #origine des rayons de la polyligne
    liste_distance_gauche = []
    liste_distance_droite = []

    for i in range(len(origines_rayons)): #pour chaque segment
        for j in range(len(origines_rayons[i][1])): #pour chaque origine associee a ce segment
            intersection_gauche, intersection_droite = intersection_rayon_batiments(origines_rayons[i][1][j], origines_rayons[i][0], batiments_candidats) #calcul du point d'intersection avec batiments candidats
            val_nulle = [0,0,0]
            distance_gauche = distance_points(intersection_gauche, origines_rayons[i][1][j]) #distance entre l'origine et l'intersection gauche
            distance_droite = distance_points(intersection_droite, origines_rayons[i][1][j]) #distance entre l'origine et l'intersection droite

            if intersection_gauche == val_nulle: #si il n'y a pas d'intersection a gauche
                val_gauche = 0 #le pixel prend la valeur 0
            else:
                val_gauche = distance_gauche #sinon il prend la valeur de la distance entre l'origine et l'intersection

            if intersection_droite == val_nulle: #si il n'y a pas d'intersection a droite
                val_droite = 0 #le pixel prend la valeur 0
            else:
                val_droite = distance_droite #sinon il prend la valeur de la distance entre l'origine et l'intersection

            liste_distance_gauche.append(val_gauche)
            liste_distance_droite.append(val_droite)
    
    return liste_distance_gauche, liste_distance_droite, nb_colonnes


# -- PARTIE 5 : CREATION DE L'IMAGE DE SORTIE --

def create_distance_array(liste_distance, nb_colonnes, nb_lignes, gauche):
    """rearengement des valeurs de distance et construction du tableau avec la bonne structure. Il est inverse si il concerne le cote droit de la rue.

    Args:
        liste_distance (_list_): liste des valeurs de distance
        nb_colonnes (_int_): nombre de colonnes dans le tableau
        nb_lignes (_int_): nombre de lignes dans le tableau
        gauche (_bool_): "vrai" si le tableau concerne le cote gauche de la rue

    Return : distance_array (_array_): matrice des distances
    """

    ligne_temp = []
    tri_1 = []
    tri_final = []
    compteur_colonne = 0
    compteur_i = 1

    while compteur_colonne < nb_colonnes: #division de la liste en n sous-liste avec n = nb_colonnes
        i_dep = int(compteur_colonne * nb_lignes)
        i_fin = int(i_dep + nb_lignes)
        ligne_temp = liste_distance[i_dep:i_fin]
        tri_1.append(ligne_temp)
        ligne_temp = []
        compteur_colonne = compteur_colonne + 1

    if gauche: #si on traite les intersections du cote gauche de la route
        for i in range(len(tri_1[0])):
            for j in range(len(tri_1)):
                ligne_temp.append(tri_1[j][-compteur_i]) #dans chaque sous-liste, recuperation des valeurs en partant de la fin
            tri_final.append(ligne_temp)
            ligne_temp = []
            compteur_i = compteur_i + 1
    else: #si on traite les intersections du cote droit de la route
        for i in range(len(tri_1[0])):
            for j in range(len(tri_1)):
                ligne_temp.append(tri_1[j][compteur_i-1]) #dans chaque sous-liste, recuperation des valeurs en partant du debut (image inversee)
            tri_final.append(ligne_temp)
            ligne_temp = []
            compteur_i = compteur_i + 1
        
    distance_array = array(tri_final)

    return distance_array

def matrice_distance(liste_polygone, route, dist_buffer, precision_min, hauteur_max):
    """calcul de la matrice des distances entre origines et intersections

    Args:
        liste_polygone (_list_): liste des polygones correspondant à tous les batiments du quartier/bbox
        route (_polyligne_): liste des points qui composent la route etudiee
        dist_buffer (_float_): distance entre la route et le batiment en dessous de laquelle le batiment sera pris en compte dans le calcul de silhouette (=buffer)
        precision_min (_float_): precision minimale (en metres). Sera equivalent a la resolution de la silhouette de sortie sur l'axe des z et des x.
        hauteur_max (_float_): hauteur maximale jusqu'a laquelle les points d'origines seront calcules. Elle est exprimee en hauteur relative par rapport a la route.

    Return : liste_pixels_gauche, liste_pixels_droite (_list_): deux listes, contenant respectivement la valeur du pixel gauche et du pixel droite 
    """

    batiments_candidats = liste_bat_candidats(liste_polygone, route, dist_buffer) #recuperation des batiments candidats
    liste_distance_gauche, liste_distance_droite, nb_colonnes = liste_distance(batiments_candidats, route, precision_min, hauteur_max) #generation des pixels pour chaque cote de la route

    return liste_distance_gauche, liste_distance_droite, nb_colonnes

def max_array(array):
    """calcul de la valeur max d'un tableau

    Args:
        array (_array_): tableau de valeur

    Return : max (_float_): la valeur maximale du tableau
    """
    max = -1
    li = array.shape[0]
    col = array.shape[1]
    
    for l in range (li):
        for c in range (col):
            d = array[l][c]
            if d > max:
                max = d

    return(max)

def conversion(array, maximum, nb_colonnes, nb_lignes):
    """conversion des valeurs de distance en pixel (de 0 a 255)

    Args:
        array (_array_): tableau de valeurs de distance
        maximum (_float_): maximum des valeurs
        minimum (_float_): minimum des valeurs

    Return : array (_array_): tableau de valeur de pixels
    """

    li = int(nb_lignes)
    col = int(nb_colonnes)
    
    pA = (0, 0)
    pB = (maximum, 255)
    a, b = calc_equa_droite(pA, pB) #calcul de la fonction affine qui a x associe sa valeur de pixel
    
    for l in range (li):       
        for c in range (col):
            curr = array[l][c]
            val_pixel = round(a*curr + b)
            array[l][c] = val_pixel #remplacement de la valeur dans le tableau d'origine
                
    return(array)

def silhouette_tab_pixel(batiments, rue, buffer, precision_min, hauteur_max):
    """calcul du tableau de pixel

    Args:
        liste_polygone (_list_): liste des polygones correspondant à tous les batiments du quartier
        route (_polyligne_): liste des points qui composent la route etudiee
        dist_buffer (_float_): distance entre la route et le batiment en dessous de laquelle le batiment sera pris en compte dans le calcul de silhouette (=buffer)
        precision_min (_float_): precision minimale (en metres). Sera equivalent a la resolution de la silhouette de sortie sur l'axe des z et des x.
        hauteur_max (_float_): hauteur maximale jusqu'a laquelle les points d'origines seront calcules. Elle est exprimee en hauteur relative par rapport a la route.

    Return : pixel_array_left, pixel_array_right 
    """

    silhouette_gauche, silhouette_droite, nb_colonnes = matrice_distance(batiments, rue, buffer, precision_min, hauteur_max) #calcul de la matrice des distances
    len_tab = len(silhouette_gauche)
    nb_lignes = len_tab/nb_colonnes

    pixel_array_left = create_distance_array(silhouette_gauche, nb_colonnes, nb_lignes, True) #creation du tableau de distance a gauche de la route
    pixel_array_right = create_distance_array(silhouette_droite, nb_colonnes, nb_lignes, False) #creation du tableau de distance a droite de la route

    max_array_left = max_array(pixel_array_left) # recuperation des max de chaque tableau
    max_array_right = max_array(pixel_array_right)
    max_value = max(max_array_left, max_array_right) #recuperation des valeurs max, tous tableaux confondus, permet d'obtenir une meme valeur de pixel pour une meme distance quelque soit le cote de la route 

    final_array_left = conversion(pixel_array_left, max_value, nb_colonnes, nb_lignes) #conversion des distances en pixels
    final_array_right = conversion(pixel_array_right, max_value, nb_colonnes, nb_lignes) #conversion des distances en pixels

    return final_array_left, final_array_right


# -- CALCUL DE LA SILHOUETTE --

def calcul_silhouette_url(bbox, rue, file_path, buffer, precision_min, hauteur_max):
    """ calcul de silhouette a partir du servce wfs de l'ign (bd topo - batiments)

    Args:
        bbox (_string_): emprise de travail, format longitude latitude : <lat_a,long_a,lat_b,long_b>
        rue (_list_): polyligne de la rue etudiee
        file_path (_string_): chemin du dossier d'ecriture des images de sortie et nom du fichier (sans extension)
        buffer (_float_): distance entre la route et le batiment en dessous de laquelle le batiment sera pris en compte dans le calcul de silhouette (=buffer)
        precision_min (_float_): precision minimale (en metres). Sera equivalent a la resolution de la silhouette de sortie sur l'axe des z et des x.
        hauteur_max (_float_): hauteur maximale jusqu'a laquelle les points d'origines seront calcules. Elle est exprimee en hauteur relative par rapport a la route.
    """

    url_bat = "https://wxs.ign.fr/topographie/geoportail/wfs?"
    url_bat  += "service=WFS&version=2.0.0&request=GetFeature&"
    url_bat += "typeName=BDTOPO_V3:batiment&"
    url_bat  += "&srsName=EPSG:2154&outputFormat=json&"

    liste_bat = get_bat_feature_from_url(url_bat, bbox) # recuperation des polygones batiment contenues dans la bbox
    silhouette_gauche, silhouette_droite = silhouette_tab_pixel(liste_bat, rue, buffer, precision_min, hauteur_max)

    cv2.imwrite(file_path + "_s_gauche.jpg", silhouette_gauche)
    cv2.imwrite(file_path + "_s_droite.jpg", silhouette_droite)

    return "Export réussi"

def calcul_silhouette_json(json_file, rue, file_path, buffer, precision_min, hauteur_max):
    """ calcul de silhouette a partir d'un fichier json (des batiments)'

    Args:
        json_file (_string_): chemin du json
        rue (_list_): polyligne de la rue etudiee
        file_path (_string_): chemin du dossier d'ecriture des images de sortie et nom du fichier (sans extension)
        buffer (_float_): distance entre la route et le batiment en dessous de laquelle le batiment sera pris en compte dans le calcul de silhouette (=buffer)
        precision_min (_float_): precision minimale (en metres). Sera equivalent a la resolution de la silhouette de sortie sur l'axe des z et des x.
        hauteur_max (_float_): hauteur maximale jusqu'a laquelle les points d'origines seront calcules. Elle est exprimee en hauteur relative par rapport a la route.
    """

    liste_bat = get_feature_from_json(json_file)
    silhouette_gauche, silhouette_droite = silhouette_tab_pixel(liste_bat, rue, buffer, precision_min, hauteur_max)

    cv2.imwrite(file_path + "_s_gauche.jpg", silhouette_gauche)
    cv2.imwrite(file_path + "_s_droite.jpg", silhouette_droite)

    return None



rue_cujas = [[651673.60003869002684951, 6861127.70463849976658821, 46.20000000000000284], [651674.5000386800384149, 6861127.10463850013911724, 46.20000000000000284],
[651744.7000386300496757, 6861064.2046385295689106, 51.5]]

rue_pantheon = [[652091.426, 6860919.774, 57.39999999999999858], [651971.673, 6860960.661, 57.39999999999999858]]

bbox = '48.84612157576351,2.339629495901875,48.84813214304847,2.3464014719858546' #emprise géographique
bbox_cassette = '48.85128,2.33032,48.84810,2.33136'

json_file = 'C:/Users/Alexandre/Documents/ENSG/M1/projet_info/pi_silhouette_de_l_espace/batiment.json'


#test_url = calcul_silhouette_url(bbox, rue_cujas, ""C://Users//Alexandre//Documents//ENSG//M1//projet_info//pi_silhouette_de_l_espace//sortie//rue_cujas", 40, 1, 25)

test_json = calcul_silhouette_json(json_file, rue_pantheon, "C://Users//Alexandre//Documents//ENSG//M1//projet_info//pi_silhouette_de_l_espace//sortie//rue_pantheon_10cm", 100, 0.1, 55)