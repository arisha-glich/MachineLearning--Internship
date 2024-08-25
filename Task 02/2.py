from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.tree import export_graphviz

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)

# Visualize the Decision Tree
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=iris.feature_names,  
                           class_names=iris.target_names,  
                           filled=True, rounded=True,  
                           special_characters=True)  

graph = graphviz.Source(dot_data)  
graph.render("decision_tree")  # Saves the visualization as a PDF file
graph.view()  # Opens the PDF file to view the tree
