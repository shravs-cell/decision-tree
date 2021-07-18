#import libraries 
import sklearn.datasets as datasets
import pandas as pd

# load dataset iris
iris=datasets.load_iris()

#forming dataframe
df=pd.DataFrame(iris.data, columns=iris.feature_names)

#forming target or the class it belongs too (species)
y=iris.target

#decision tree algorithm 
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(df,y)

#import libraries for showing the graph
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
#export_graphviz doesn't work till you specify path sometimes 
import os
os.environ["PATH"] += os.pathsep +'C:\\Users\\username\\Anaconda3\\Library\\bin\\graphviz'
import pydotplus

#visualizing graph for decision tree algorithm 
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, 
                rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
