"""
    Decision Tree Builder created using chess dataset. ID3 algorithm is used for building the tree.
    Programming Languages: Python
    Author: Aishwarya Bhosale,Apoorva Uppala, Dinesh Reddy Kommera, Keshav Bhojanapally
"""
import pandas as pd
import numpy as np
from pprint import pprint
import random
random.seed(10)

total_class_labels = 18 #Total Number of class labels in the dataset
#Function for calculating the entropy for the target column
def calculate_entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)

    entropy = np.sum([(-counts[i]/np.sum(counts))*(np.log2(counts[i]/np.sum(counts))/np.log2(total_class_labels)) for i in range(len(elements))])
    return entropy

#Function for finding the information gain for each attribute.
def information_gain(data,split_attribute_name,target_name="class"):
       
    total_entropy = calculate_entropy(data[target_name],)

    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*calculate_entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    Information_Gain = total_entropy - Weighted_Entropy
    return (Information_Gain,Weighted_Entropy)


# the main function for generating the tree which will be called recursively
def generating_tree(data,originaldata,features,target_attribute_name="class",parent_node_class = None):
    #stopping criteria
    
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    
    elif len(features) ==0:
        return parent_node_class
        
    else:
        #expanding the tree
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        item_values = [information_gain(data,feature,target_attribute_name)[0] for feature in features] #Return the information gain values for the features in the dataset
        entropy_values=[information_gain(data,feature,target_attribute_name)[1] for feature in features] 
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature:{}}
        
        print("Best Feature: ", best_feature, "Information Gain: ",item_values[best_feature_index],"Entropy: ",entropy_values[best_feature_index])

        features = [i for i in features if i != best_feature]
                
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            #generating the subtree.
            subtree = generating_tree(sub_data,dataset,features,target_attribute_name,parent_node_class)
            print("Parent: ",best_feature,value)
            #assigning the value to the tree.
            tree[best_feature][value] = subtree
            
        return(tree)    
                 
#function for predicting the value of the query depending on the trained tree.    
def predict(query,tree,default = 1):
    
    
    #1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return predict(query,result)

            else:
                return result
            
#function for testing and finding the mis-classed vectors
def find_accuracy(data,tree):
    mis_classed_vectors=pd.DataFrame()
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0)
        if(predicted["predicted"][i]!=data["class"][i]):
            mis_classed_vectors = mis_classed_vectors.append(data.iloc[i],ignore_index=True)
    #print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["class"])/len(data))*100,'%')
    accuracy = (np.sum(predicted["predicted"] == data["class"])/len(data))*100
    return mis_classed_vectors,accuracy
    
def find_holdout(training_data, holdout_data,mis_classed_vectors):
    final_training_data = []
    final_holdout_data = []
    
    Training_indexes = list(training_data.index)
    Testing_indexes = list(holdout_data.index)
    union_data=Training_indexes+Testing_indexes
    union_data.sort()
    
    for i in mis_classed_vectors:
        union_data.append(i)
        union_data.append(i)

    for _ in range(len(training_data)):
        add_index = random.randint(0, len(training_data) - 1)
        final_training_data.append(union_data[add_index])



    # remove duplicates before removing items in final_t_set
    for item in union_data:
        if item not in final_holdout_data:
            final_holdout_data.append(item)
    
    for item in final_training_data:
        if item in final_holdout_data:
            final_holdout_data.remove(item)


    return final_training_data, final_holdout_data
    
def convert_to_dataframe(training_set):
    
    training_set.sort()   
    data=[]
    for i, j in dataset.iterrows():
        if i in training_set:
            count=training_set.count(i)
            for _ in range(count):
                data.append(dataset[i:i+1].values)

    temp_list_1=[]
    for i in data:
        temp_list_2=[]
        for t in i:
            for r in t:
                temp_list_2.append(r)
        temp_list_1.append(temp_list_2)
    return temp_list_1
    
print("*************************FIRST DTREE***************************")
dataset = pd.read_csv('550-p1-cset-krk-1.csv', header=None, names=['WKf', 'WKr', 'WRf', 'WRr', 'BKf', 'BKr','class'])
independent = dataset.drop('class', axis=1) #dropping the class lable for result class
independent = pd.get_dummies(independent.astype(str))
dataset=pd.concat([independent,dataset['class']],axis=1)

training_data_1, holdout_data_1, validation_data = np.split(dataset.sample(frac=1,random_state=42), [int(.6*len(dataset)), int(.8*len(dataset))])

training_data_1=training_data_1.reset_index(drop=True)
holdout_data_1=holdout_data_1.reset_index(drop=True)
validation_data=validation_data.reset_index(drop=True)

print("*********************TRAINING DATA****************************")
print(training_data_1)
print("**********************HOLDOUT SET*****************************")
print(holdout_data_1)
print("*******************VALIDATION DATA*****************************")
print(validation_data)


dTree_1 = generating_tree(training_data_1,training_data_1,training_data_1.columns[:-1])


print('********************Class for each leaf node**********************')
pprint(dTree_1)
mis_classed_vectors_1,dTree1_accuracy=find_accuracy(holdout_data_1,dTree_1)

training_data_2,holdout_data_2 = find_holdout(training_data_1,holdout_data_1,mis_classed_vectors_1)

#Second Tree
print("***********************SECOND DTREE***************************")

training_data_2 = pd.DataFrame(data= convert_to_dataframe(training_data_2),columns=dataset.columns)
holdout_data_2 = pd.DataFrame(data=  convert_to_dataframe(holdout_data_2[:len(holdout_data_2)-41]),columns=dataset.columns)
print("*********************TRAINING DATA****************************")
print(training_data_2)
print("**********************HOLDOUT SET*****************************")
print(holdout_data_2)
print("*******************VALIDATION DATA*****************************")
print(validation_data)

dTree_2 = generating_tree(training_data_2,training_data_2,training_data_2.columns[:-1])
print('********************Class for each leaf node**********************')

pprint(dTree_2)
mis_classed_vectors_2,dTree2_accuracy=find_accuracy(holdout_data_2,dTree_2)

print('Accuracy of Dtree 1:',dTree1_accuracy)
print('Accuracy of Dtree 2:',dTree2_accuracy)

print('********************Mis-classed Holdout Vectors DTree1********************')
print(mis_classed_vectors_1)

print('********************Mis-classed Holdout Vectors DTree2********************')
print(mis_classed_vectors_2)

print('********************Validation Set Accuracy DTree1********************')
mis_classed_vectors,dTree1_validation_accuracy=find_accuracy(validation_data,dTree_1)
print('Accuracy of DTree 1 on validation set:',dTree1_validation_accuracy)

print('********************Validation Set Accuracy DTree2********************')
mis_classed_vectors,dTree2_validation_accuracy=find_accuracy(validation_data,dTree_2)
print('Accuracy of DTree 2 on validation set:',dTree2_validation_accuracy)      