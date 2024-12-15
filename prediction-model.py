#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


# In[2]:


# Path to your file
file = "merged_filled_data.xlsx"


# In[3]:


# Read the Excel file in
data = pd.read_excel(file)


# In[4]:


print(data.head())


# In[5]:


# Select the row with ID 'Q8TD30'
row_with_id = data[data['ID'] == 'Q8TD30']


# In[6]:


print(row_with_id)


# In[7]:


# Save this row to a new file
output_file_path = "Q8TD30_rowmerged.xlsx"
row_with_id.to_excel(output_file_path, index=False)


# In[8]:


print(f"The row with ID 'Q8TD30' is stored in {output_file_path}")


# In[9]:


# Path to your file
filex = "Q8TD30_rowmerged.xlsx"


# In[10]:


# Read the Excel file in
datax = pd.read_excel(filex)


# In[11]:


print(datax)


# In[12]:


# Path to your file
filemeta = "metadata.xlsx"


# In[13]:


# Read the Excel file in
metadata = pd.read_excel(filemeta)


# In[14]:


print(metadata.head())


# In[15]:


# Select columns 'ID' and 'Proteomic_Subtype'
filtered_metadata = metadata[["ID", "Proteomic_Subtype"]]


# In[16]:


print(filtered_metadata)


# In[17]:


# Filter patients in metadata based on similarity to column names in data
metadata["Patient_ID"] = metadata["ID"].str[:4] # The first 4 characters of the patient ID
data_columns_subset = [col for col in datax.columns if col[:4] in metadata["Patient_ID"].values]


# In[18]:


# Add proteomic subtypes to the bottom of the dataset
metadata_dict = metadata.set_index("Patient_ID")["Proteomic_Subtype"].to_dict()  # Map patients by subtypes
proteomic_subtypes = [metadata_dict[col[:4]] if col[:4] in metadata_dict else "-" for col in datax.columns]


# In[19]:


# Create a new dataset with protein abundances and proteomic subtypes
combined_data = pd.DataFrame(
    data=datax.values,
    index=datax.index,
    columns=datax.columns,
)
combined_data.loc["Proteomic_Subtype"] = proteomic_subtypes # Add the subtypes as the last row


# In[20]:


print(combined_data)


# In[21]:


# Save the combined data as an Excel file
output_file_path = "merged_data.xlsx"
combined_data.to_excel(output_file_path, index=True)


# In[22]:


print(f"Combined data stored in: {output_file_path}")


# In[ ]:





# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# In[2]:


# Load the data
file = "merged_data.xlsx"
combined_data = pd.read_excel(file, index_col=0)


# In[3]:


print(combined_data.head())


# In[4]:


# Step 1: Retrieve protein abundances (X) and labels (y)
X = combined_data.iloc[0, 3:].values.astype(float)  # Abundances of row 0
y = combined_data.iloc[1, 3:].values               # Labels (Proteomic_Subtype) from row 1


# In[5]:


# Check shapes
print("Shape of X (features):", X.shape)
print("Shape of y (labels):", y.shape)


# In[6]:


# Normalize data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X.reshape(-1, 1))  # Correct transformation


# In[7]:


# Train-test-splitsing (stratify=y)
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.3, random_state=42, stratify=y
)


# In[8]:


# Check forms of splits
print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)


# In[9]:


# Using Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[10]:


# Model testing on the test set
y_pred = clf.predict(X_test)
print("Model performance on the test set:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


# In[13]:


# Function to make predictions for a specific patient
def predict_proteomic_subtype(patient_id, combined_data, model, scaler):
    # Check if the patient is in the dataset
    if patient_id not in combined_data.columns:
        print(f"Patient {patient_id} not found in the dataset.")
        return
    # Retrieve abundances for the patient
    patient_data = combined_data[patient_id].iloc[:-1].values.astype(float)  # Skip last row (Proteomic_Subtype)
    patient_data_normalized = scaler.transform(patient_data.reshape(1, -1))
    
    # Making predictions
    prediction = model.predict(patient_data_normalized)
    print(f"Predicted proteomic subtype for patient {patient_id}: {prediction[0]}")


# In[14]:


# Making predictions for a specific patient
patient_id = input("Enter patient ID: ")
predict_proteomic_subtype(patient_id, combined_data, clf, scaler)


# In[15]:


from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# In[16]:


# Select the first tree from the RandomForestClassifier
tree = clf.estimators_[0]

# Make sure class_names is a list
class_names = clf.classes_  # Haal de klassen op
print("Type of class_names:", type(class_names))  # Debugging to see what it is

# Make sure it's a list
class_names = list(class_names)
print("Converted class_names:", class_names)  # Debugging to see what we pass on

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=["Protein Abundance"], class_names=class_names, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree uit Random Forest")
plt.show()


# In[17]:


#Code for generating an ROC curve:''


# In[18]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# In[19]:


# Obtain the predictions of the probabilities of each class
y_prob = clf.predict_proba(X_test)


# In[20]:


# Binarize the real classes (convert to an array of 0 and 1)
y_test_bin = label_binarize(y_test, classes=clf.classes_)


# In[21]:


# Number of classes
n_classes = len(clf.classes_)


# In[22]:


# Create an empty figure for the ROC curve
plt.figure(figsize=(10, 8))


# In[23]:


# Plot the ROC curve for each class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)  # Calculate the AUC for this class
    plt.plot(fpr, tpr, lw=2, label=f'Class {clf.classes_[i]} (AUC = {roc_auc:.2f})')
# Add the 'diagonal' line for arbitrary classification
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# Add labels and title to the chart
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for multiclass classification.')
plt.legend(loc="lower right")

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




