# Decision-Tree-Ensemble

 CECS550

# Project1 - DTrees Ensemble

# Team: Incognito

# Team Members:
  Aishwarya Bhosale
  Apoorva Uppala
  Dinesh Reddy Kommera
  Keshav Bhojanapally
  
# Intro:
  Ensemble methods, which combines several decision trees to produce better predictive performance than utilizing a single decision tree. The main principle behind the ensemble model is that a group of weak learners come together to form a strong learner.
  Description
  * In this project, we used ID3 algorithm for creating a automated decision tree builder. For this builder, we created two decision trees using classified set of feature vectors.
  * We have used the 550-p1-cset-krk-1.csv sample dataset for building the decision tree.
  * First Decision tree has been created with random selection of training set, holdout set.
  * Second Decision tree has been created with union of first training set and mis classed holdout vectors.
  * We utilized the Shannon-based Entropy and Information Gains for finding the best split in the tree.

# Contents:
  550-01-p1-Incognito.zip consists of CECS550_Project1.py, 550-p1-cset-krk-1.csv and Readme.txt
  
# Setup and Installation:
  * Navigate to python downloads page and install python with specified steps.
  * pip install pandas
  * pip install numpy
  * pip install random
  * pip install pprint
  
# Sample Invocation:
  python CECS550_Project1.py

  
# References:
  https://www.python-course.eu/Boosting.php
  https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification
  https://www.python-course.eu/Decision_Trees.php
