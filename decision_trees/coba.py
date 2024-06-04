import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('Fish.csv')

# Function to calculate entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# Function to calculate information gain
def info_gain(data, split_attribute_name, target_name="Species"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Function to create the decision tree
def decision_tree(data, original_data, features, target_attribute_name="Species", parent_node_class=None):
    # If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    
    # If the feature space is empty, return the parent node class
    elif len(features) == 0:
        return parent_node_class
    
    # If none of the above holds true, grow the tree
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        tree = {best_feature: {}}
        
        features = [i for i in features if i != best_feature]
        
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            
            subtree = decision_tree(sub_data, original_data, features, target_attribute_name, parent_node_class)
            
            tree[best_feature][value] = subtree
            
        return tree

# Function to predict the class of a single data point
def predict(tree, sample):
    for nodes in tree.keys():
        value = sample[nodes]
        tree = tree[nodes][value]
        
        if isinstance(tree, dict):
            prediction = predict(tree, sample)
        else:
            prediction = tree
            break
    return prediction

# Define the feature names
features = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]

# Build the decision tree
tree = decision_tree(data, data, features)
# print("Decision Tree: ", tree)

# Function to take user input and make a prediction
def classify_fish():
    weight = float(input("Enter Weight: "))
    length1 = float(input("Enter Length1: "))
    length2 = float(input("Enter Length2: "))
    length3 = float(input("Enter Length3: "))
    height = float(input("Enter Height: "))
    width = float(input("Enter Width: "))

    sample = {
        "Weight": weight,
        "Length1": length1,
        "Length2": length2,
        "Length3": length3,
        "Height": height,
        "Width": width
    }

    prediction = predict(tree, sample)
    print(f"The predicted species of the fish is: {prediction}")

# Run the classification function
classify_fish()