# Exercice d'évaluation pour le poste Inria / AP-HP

### data.db

Base de données pour l'étude.

### data_issues.ipynb

Présente les problèmes de qualité de la base de données.
- valeurs manquantes
- types des colonnes
- information absente mais déductible (mais pas toujours vraie)
- echange de colonnes
- doublons
- incohérences

### detect_duplicate.py

Adapté d'un fuzzy matching.

Une fonction 'correct_state' pour nettoyer les typos de 'state'

3 fonctions pour la détection de doublons : ngrams, awesome_cossim_top, detect_duplicates
Filtrage des doublons (avec typos/ valeurs manquantes/ échange de colonnes) par calcul de TF-IDF


### EDA.ipynb

-Présentation des données
-Données manquantes
-Nettoyage des données
-Résultats de tests pcr

### test_detect_duplicate.py

Tests sur différents cas de doublons avec pytest
- un ou deux NAN (le test réduit les doublons de 12 à 2 donc un en trop)
- les typos (test réussi)
- les échanges de colonnes (test réussi)

### Librairies

-pandas
-sqlalchemy
-numpy
-matplotlib
-seaborn
-re
-sklearn
_scipy
-sparse_dot_topn
-enchant
-pytest
