import matplotlib.pyplot as plt 
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, ConfusionMatrixDisplay

from sklearn import tree
#************************************************************************************
from sklearn import preprocessing

#************************************************************************************

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster      import KMeans
from sklearn.tree         import DecisionTreeRegressor
from sklearn.tree         import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#************************************************************************************
#preprocessing
df = pd.read_csv('H:\\Programming AI\\مجلد للتطبيق و التجريب\\Decision-Tree-Classifier\\bill_authentication.csv')

# Split features and labels
X=df.drop('Class', axis=1)
y=df['Class']
#************************************************************************************
# Train test split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.9, random_state=13)

# Decision Tree Model
# Train the model

model = DecisionTreeClassifier(
    criterion = 'gini',
    splitter = 'best'
)
model.fit(X_train, y_train)

#************************************************************************************

# Tree plotting


plt.figure(figsize=(40,30))
tree.plot_tree(
    model,
    feature_names=['Variance','Skewness', 'Curtosis', 'Entropy'], 
    class_names = ['Not authentic','Authentic'],
    filled = True
)
plt.show()

#************************************************************************************
# Model prediction
y_pred = model.predict(X_test)
#Model evaluation
acc = accuracy_score(y_pred,y_test)
cm = confusion_matrix(y_test,y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
# Demo the evaluation
print('Accuracy:',acc)
print(classification_report(y_test,y_pred))
cm_display.plot()
plt.show()