#!/usr/bin/env python
from __future__ import print_function

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

### Load training data ###
raw = pd.read_json("../Data/train.json")

### Preprocessing ###

# __Labels__
# Find all possible cuisines
posibleCuisine = set(raw["cuisine"])

#  Make map: Cuisine->{0,1,2,..}
indexToCuisine = dict(zip(range(len(posibleCuisine)), posibleCuisine))

# __Features__
# Find all possible ingredients
posibleIngredients = list(set(reduce(lambda x,y: x+y, [x for x in raw["ingredients"]])))
#ingredientToIndex = dict( zip(posibleIngredients, len(posibleIngredients)) )

#print(ingredientToIndex)


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
print(tblFeatures)



### Train ###
# Fitting the decision tree with scikit-learn
y = range(len(posibleCuisine) # Labels
X = tblFeatures # features
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)

### Visualization ###
def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

visualize_tree(dt, features)

### Load test data ###

### Predict ###

### Save submission ###

# In order to pass this data into scikit-learn we need to encode the 
# Names to integers



############################################################

#df2, targets = encode_target(df, "Name")



# Visualizing the tree:

# Make a solution submission

