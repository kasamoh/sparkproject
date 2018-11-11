
# Projet Spark-scala - Réalisé par Mohamed DHAOUI#

#### Téléchargement et importation du projet dans IntelliJ:

1. Télécharger le repository https://github.com/kasamoh/sparkproject en tant que ZIP  ( il faut cliquer sur" clone or download " puis ""download zip"
2. Extraire le fichier zip dans un repertoire de travail que vous choisissez 
3. Ouvrir l'IDE IntelliJ , click sur "File " ==> "New " ==> "Project from existing source"
4. Selectionner le dossier que vous venez d'extraire 
5. Selectionner "import project from external model" et ensuite en bas "sbt"
6. Cliquer sur "finish" puis attendre quelques secondes pour le chargement du projet

#### Modifications pour lancement :
Au niveau de la hierarchie des répertoires du projet à gauche de l'IDE ,selectionner le fichier **build_and_submit.sh** :

Il faut changer l'url path_to_spark avec le chemin du dossier **spark-2.2.0-bin-hadoop2.7**  si ce dernier ne se trouve pas dans le "HOME"
```
path_to_spark="$HOME/spark-2.2.0-bin-hadoop2.7"
```

Selectionner maintenant "src" ==> "main" ==> "scala" ==> "com.sparkProject"==> "Trainer.scala"

Afin d'exécuter le programme Trainer.scala depuis IntelliJ , il faut aller à l'onglet "terminal" et tapper : 

```
./build_and_submit.sh Trainer
```

#### Résultat : 

Le programme devrait prendre entre 5 et 7 minutes pour s'éxécuter . A la fin , on obtient le score et la la matrice de confusion ci-dessous : 

**F1 score = 0.655**


| final_status | predictions | count |
|:---------:|:-----------:|:-------:|
| 1         | 0           |   1020  |
| 0         | 1           |   2836  |
| 1         | 1           |   2446  |
| 0         | 1           |   4512  |
