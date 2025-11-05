import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import RandomForestClassifier
#eliminating run time errors
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


data = []
labels = []
path = "Brain_Tumor_Dataset"

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        data.append(img.flatten())   # flatten image → vector of pixel values
        labels.append(1 if folder == 'yes' else 0)
x = np.array(data)
y = np.array(labels)
print(f"Data shape: {x.shape}")
print(f"Labels shape: {y.shape}")

#Split and train
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(f" X_train = {x_train.shape},X_test = {x_test.shape}")

#replacing NaN or inf with 0
x_train = np.nan_to_num(x_train)
x_test = np.nan_to_num(x_test)

# Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#performing svc on raw data
clf = SVC(kernel="rbf")
clf.fit(x_train,y_train)
ac = accuracy_score(y_train,clf.predict(x_train))
print("--------------------------------")
print("SVC on unprocessed data")
print(f"Training accuracy is: {ac}")
ac = accuracy_score(y_test, clf.predict(x_test))
print(f"Testing accuracy is: {ac}")
print(f"Number of supporting vectors: {clf.support_vectors_.shape[0]}")

#The svc accuracy score 0.74 is too law, trying pca
pca = PCA(n_components=0.95)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print("Performing PCA")
print(f"x_train_pca shape: {x_train_pca.shape}, x_test_pca shape: {x_test_pca.shape}")

#performing svc on principle components
clf = SVC(kernel="rbf")
clf.fit(x_train_pca,y_train)
ac = accuracy_score(y_train,clf.predict(x_train_pca))
print("--------------------------------")
print("SVC on processed data")
print(f"Training accuracy is: {ac}")
ac = accuracy_score(y_test, clf.predict(x_test_pca))
print(f"Testing accuracy is: {ac}")
print(f"Number of supporting vectors: {clf.support_vectors_.shape[0]}")


#visualisation of principle components
pca_2d = PCA(n_components=2)
pca_2d.fit(x_train,y_train)
x_train_2d_pca = pca_2d.transform(x_train)
x_test_2d_pca = pca_2d.transform(x_test)

fig = plt.figure(figsize=(8,6))

plot = fig.add_subplot(1,2,1)
plt.scatter(x_train_2d_pca[y_train==0, 0], x_train_2d_pca[y_train==0, 1], color='blue', label='No Tumor')
plt.scatter(x_train_2d_pca[y_train==1, 0], x_train_2d_pca[y_train==1, 1], color='red', label='Tumor')
plot.set_title('PCA Projection (2D) - training set')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plot = fig.add_subplot(1,2,2)
plt.scatter(x_test_2d_pca[y_test==0, 0], x_test_2d_pca[y_test==0, 1], color='blue', label='No Tumor')
plt.scatter(x_test_2d_pca[y_test==1, 0], x_test_2d_pca[y_test==1, 1], color='red', label='Tumor')
plot.set_title('PCA Projection (2D) - test set')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

#visualising using svc
clf_2d = SVC(kernel="rbf", C=100, gamma=1)
clf_2d.fit(x_train_2d_pca, y_train)

fig, ax = plt.subplots(figsize=(8, 6))

# Plot the decision boundary
DecisionBoundaryDisplay.from_estimator(
    clf_2d,
    x_train_2d_pca,
    response_method="predict",
    cmap=plt.cm.coolwarm,
    alpha=0.5,
    ax=ax,
    xlabel="Principal Component 1",
    ylabel="Principal Component 2",
)

ax.scatter(
    x_train_2d_pca[y_train == 0, 0],
    x_train_2d_pca[y_train == 0, 1],
    c='blue',
    label='No Tumor',
    edgecolors='k',  # Add black edges for clarity
    s=40
)
ax.scatter(
    x_train_2d_pca[y_train == 1, 0],
    x_train_2d_pca[y_train == 1, 1],
    c='red',
    label='Tumor',
    edgecolors='k',
    s=40
)

ax.set_title("SVC Decision Boundary on 2D PCA Data (RBF Kernel)")
ax.legend()
plt.show()



#random forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(x_train_pca, y_train)

# Evaluate
print("\n--------------------------------")
print("Random Forest on PCA Data")
ac_train_rf = accuracy_score(y_train, rf_clf.predict(x_train_pca))
print(f"RF Training accuracy is: {ac_train_rf}")

ac_test_rf = accuracy_score(y_test, rf_clf.predict(x_test_pca))
print(f"RF Testing accuracy is: {ac_test_rf}")
