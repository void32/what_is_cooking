#!/usr/bin/env python
"""
python decision_tree.py [min_samples_split for each worker thread]

Example:
    python decision_tree.py 1000 2000 5000

    Produces thread with a decision tree for 1000, 2000, and 5000.

"""
from __future__ import print_function
import sys

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import threading

### Load training data ###
raw = pd.read_json("../Data/train.json")

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

print("#posibleIngredients="+str(len(posibleIngredients)))

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

################################################################################

class WorkerThread(threading.Thread):

    def __init__(self, min_samples_split):
        threading.Thread.__init__(self)
        self.mss = int(min_samples_split)
        self.dt = None
        self.testID = None

    def train(self):
        ### Train ###
        # Fitting the decision tree with scikit-learn
        y = tblLabels # Labels
        X = tblFeatures # features

        # Train the model
        self.dt = DecisionTreeClassifier(min_samples_split=self.mss, random_state=99)
        self.dt.fit(X, y)

        ### Visualization ###
        def visualize_tree(tree, feature_names, name_suffix=""):
            with open("../Visualizations/dt"+str(name_suffix)+".dot", 'w') as f:
                export_graphviz(tree, out_file=f,
                                feature_names=feature_names)

            command = ["dot", "-Tpng", "../Visualizations/dt"+str(name_suffix)+".dot", "-o", "../Visualizations/dt"+str(name_suffix)+".png"]
            try:
                subprocess.check_call(command)
            except:
                exit("Could not run dot, ie graphviz, to "
                     "produce visualization ("+name_suffix+")")

        visualize_tree(self.dt, posibleIngredients, str(self.mss))



    def predict(self):
        if self.dt != None:
            ### Load test data ###
            test = pd.read_json("../Data/test.json")
            self.testID = test["id"]
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
            predictedCuisineIndex = self.dt.predict(testTblFeatures)
            self.predictedCuisineName = [indexToCuisine[i] for i in predictedCuisineIndex]
        else:
            print("self.dt="+str(self.dt))
            print("self.testID="+str(self.testID))

    def output(self):
        ### Write to submission file ###
        submissionFile = open('../Submissions/submission'+str(self.mss)+'.csv','w')
        submissionFile.write("id,cuisine\n")
        for i in range(len(self.testID)):
            submissionFile.write(str(self.testID[i])+","+str(self.predictedCuisineName[i])+"\n")
        submissionFile.close()

    def run(self):
        print("Starting " + str(self.mss))
        self.train()
        self.predict()
        self.output()
        print("Exiting " + str(self.mss))


################################################################################

# Get the min_samples_split as program arguments
for mss in sys.argv[1:]:
    if int(mss) > 0:
        # Create new threads
        thread = WorkerThread(int(mss))
        
        # Start thread
        thread.start()
    else:
        print("min_samples_split must be greater than zero: mss="+str(mss))

print("Exiting Main Thread")


















