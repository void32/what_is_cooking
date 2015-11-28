#!/usr/bin/env python
from __future__ import print_function
import sys

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

### Load training data ###
raw = pd.read_json("../Data/small_train.json")

### Preprocessing ###

# __Labels__
# Find all possible cuisines
posibleCuisine = list(set(raw["cuisine"]))
print("#posibleCuisine="+str(len(posibleCuisine)))
#  Make map: Cuisine->{0,1,2,..}
indexToCuisine = dict(zip(range(len(posibleCuisine)), posibleCuisine))

tblLabels = [posibleCuisine.index(x) for x in raw["cuisine"]]

# __Features__
# Find all possible ingredients
posibleIngredients = list(set(reduce(lambda x,y: x+y, [x for x in raw["ingredients"]])))

print("#posibleIngredientslen="+str(len(posibleIngredients)))

# Make table with row for each training case and collumn for each possible ingredient
#{0,1},{0,1},{0,1},...{0,1}
#{0,1},{0,1},{0,1},...{0,1}
# .                        
#           .              
#                       .  
#{0,1},{0,1},{0,1},...{0,1}
tblFeatures = []
for stufs in raw["ingredients"]:
    row = []
    for colIndex in range(len(posibleIngredients)):
        if posibleIngredients[colIndex] in stufs:
            row.append(1)
        else:
            row.append(0)
    tblFeatures.append(row)



### Train ###
# Fitting the decision tree with scikit-learn
y = tblLabels # Labels
X = tblFeatures # features

# Get the min_samples_split as the no 1 argument 
if (len(sys.argv) > 1):
    mss=int(sys.argv[1]) #min_samples_split
else:
    mss=1000 # A default value

dt = DecisionTreeClassifier(min_samples_split=mss, random_state=99)
dt.fit(X, y)

### Visualization ###
def visualize_tree(tree, feature_names, name_suffix=""):
    with open("dt"+str(name_suffix)+".dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt"+str(name_suffix)+".dot", "-o", "dt"+str(name_suffix)+".png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization ("+name_suffix+")")

visualize_tree(dt, posibleIngredients, mss)

### Load test data ###
test = pd.read_json("../Data/test.json")
testID = test["id"]
#testIngredients = test["ingredients"]

# Make table with row for each training case and collumn for each possible ingredient
#{0,1},{0,1},{0,1},...{0,1}
#{0,1},{0,1},{0,1},...{0,1}
# .                        
#           .              
#                       .  
#{0,1},{0,1},{0,1},...{0,1}
testTblFeatures = []
for stufs in test["ingredients"]:
    row = []
    for colIndex in range(len(posibleIngredients)):
        if posibleIngredients[colIndex] in stufs:
            row.append(1)
        else:
            row.append(0)
    testTblFeatures.append(row)

### Predict ###
predictedCuisineIndex = dt.predict(testTblFeatures)
predictedCuisineName = [indexToCuisine[i] for i in predictedCuisineIndex]


### Write to submission file ###
submissionFile = open('submission'+str(mss)+'.csv','w')
submissionFile.write("id,cuisine\n")
for i in range(len(testID)):
    submissionFile.write(str(testID[i])+","+str(predictedCuisineName[i])+"\n")
submissionFile.close()

print("done:"+str(mss))
