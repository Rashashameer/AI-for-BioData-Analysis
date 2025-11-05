import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, cross_val_score,cross_validate

#loading the data set
data = pd.read_csv("data.csv")
print(data.head())

# Data preprocessing
x = data.drop(['Unnamed: 32','id','diagnosis'], axis = 1)
y = data['diagnosis']
print(y.value_counts())

#Encoding of y values
le = LabelEncoder()
le.fit(y)
y = le.transform(y)

# print(y[y==0].shape ,y[y==1].shape) - crosschecking the code given

#Split Dataset
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
# print(x_train.shape,x_test.shape)

# Standardising data
std = StandardScaler()
x_std = std.fit(x_train)
x_train = std.transform(x_train)
x_test = std.transform(x_test)

#KNN classification
k_fold = KFold(n_splits=5)
clf = KNeighborsClassifier()
scores = cross_validate(clf,x_train,y_train, cv = k_fold, scoring=['accuracy','precision','recall','f1'])
print(scores)

#pca
pca = PCA(n_components=2)
pca.fit(x_train)
print(pca.explained_variance_ratio_)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

#visualisation using PC-1 and PC-2
x_test_actual_B = x_test_pca[y_test==0]
x_test_actual_M = x_test_pca[y_test==1]

fig = plt.figure(figsize=(12,5))

plot = fig.add_subplot(1,2,1)
plt.scatter(x_test[:,0],x_test[:,1],color = 'cyan')
plot.set_title("Visualising without any labels")

plot = fig.add_subplot(1,2,2)
plt.scatter(x_test_actual_B[:,0],x_test_actual_B[:,1])
plt.scatter(x_test_actual_M[:,0],x_test_actual_M[:,1])
plot.set_title("Visualising with labels")
plt.legend(['B: Benign','M: Malignant'],loc ="upper right")
plt.show()

# KMeans on original data
kmeans = KMeans(n_clusters=2, random_state=0,max_iter=500).fit(x_train)
kmeans.labels_
ac = accuracy_score(y_train,kmeans.labels_)
print("Kmeans on Given data")
print(f"Training accuracy is: {ac}")
y_predicted = kmeans.predict(x_test)
ac = accuracy_score(y_test, y_predicted)
print(f"Testing accuracy is: {ac}")

#Visualisation
x_test_predicted_B=x_test_pca[y_predicted==0]
x_test_predicted_M=x_test_pca[y_predicted==1]

x_test_actual_B=x_test_pca[y_test==0]
x_test_actual_M=x_test_pca[y_test==1]

fig = plt.figure(figsize=(12,5))

plot = fig.add_subplot(1,2,1)
plt.scatter(x_test_predicted_B[:,0],x_test_predicted_B[:,1],color = 'green')
plt.scatter(x_test_predicted_M[:,0],x_test_predicted_M[:,1],color = 'red')
plt.xlabel('PC-1')
plt.ylabel('PC-2')
plot.set_title("Output obtained from Kmeans clustering")
plt.legend(['B: Benign','M: Malignant'],loc ="upper right")

plot=fig.add_subplot(1,2,2)
plt.scatter(x_test_actual_B[:,0],x_test_actual_B[:,1],color = 'green')
plt.scatter(x_test_actual_M[:,0],x_test_actual_M[:,1],color = 'red')
plt.xlabel('PC-1')
plt.ylabel('PC-2')
plt.legend(['B: Benign','M: Malignant'],loc ="upper right")
plot.set_title(f"Given data")
plt.show()

#Kmeans on Principal components
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto", max_iter=500).fit(x_train_pca)
kmeans.labels_
ac=accuracy_score(y_train, kmeans.labels_)
print("Kmeans on principal components")
print(f"Training accuracy is: {ac}")
y_predicted = kmeans.predict(x_test_pca)
ac = accuracy_score(y_test,y_predicted)
print(f"Testing accuracy is: {ac}")

#visualisation
x_test_predicted_B = x_test_pca[y_predicted==0]
x_test_predicted_M = x_test_pca[y_predicted==1]

X_test_actual_B = x_test_pca[y_test==0]
X_test_actual_M = x_test_pca[y_test==1]

fig = plt.figure(figsize=(12,5))

plot = fig.add_subplot(1,2,1)
plt.scatter(x_test_predicted_B[:,0],x_test_predicted_B[:,1], color = 'pink')
plt.scatter(x_test_predicted_M[:,0],x_test_predicted_M[:,1], color = 'purple')
plt.xlabel("PC-1")
plt.ylabel("PC-2")
plot.set_title("Kmeans on principle components")

plot = fig.add_subplot(1,2,2)
plt.scatter(x_test_actual_B[:,0],x_test_actual_B[:,1], color = 'pink')
plt.scatter(x_test_actual_M[:,0],x_test_actual_M[:,1], color = 'purple')
plot.set_title("Given data")
plt.xlabel("PC-1")
plt.ylabel("PC-2")
plt.show()

#SVM using first two principle components
clf = SVC(kernel="linear",C =1000)
clf.fit(x_train_pca, y_train)

ac = accuracy_score(y_train,clf.predict(x_train_pca))
print(f"Training accuracy is: {ac}")
ac = accuracy_score(y_test, clf.predict(x_test_pca))
print(f"Testing accuracy is: {ac}")
print(f"number of support vectors: {clf.support_vectors_.shape[0]}")
plt.scatter(x_train_pca[:, 0][y_train==0], x_train_pca[:, 1][y_train==0], s=30, )
plt.scatter(x_train_pca[:, 0][y_train==1], x_train_pca[:, 1][y_train==1], s=30, )

plt.legend(["B","M"])
ax = plt.gca()

#plotting the decision boundary
DecisionBoundaryDisplay.from_estimator(
    clf,
    x_train_pca,
    plot_method ="contour",
    colors ="k",
    levels =[-1, 0, 1],
    alpha =0.5,
    linestyles =["--", "-", "--"],
    ax=ax,
)
#plotting the support vectors
ax.scatter(
    clf.support_vectors_[:,0],
    clf.support_vectors_[:,1],
    s =100,
    linewidth=0.5,
    facecolors = "none",
    edgecolors= "k",
)
plt.xlabel('PC-1')
plt.ylabel('PC-2')
plt.title('Decision Boundary Display for SVM (linear kernel function) ')
plt.show()

#Number of training samples vs Test accuracies
train_sizes = []
test_accuracies = []
train_accuracies = []

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

std = StandardScaler()
x_std = std.fit(x_train)
x_train = x_std.transform(x_train)
x_test = x_std.transform(x_test)

for train_size in range(10, 410, 10):
    x_train_1 = x_train[:train_size]
    y_train_1 = y_train[:train_size]

    clf = SVC(gamma='auto')
    clf.fit(x_train_1, y_train_1)
    ac = accuracy_score(y_test, clf.predict(x_test))
    print('Train size:', train_size)
    print('Test accuracy:', ac)
    print('')
    train_sizes.append(train_size)
    test_accuracies.append(ac)
    train_accuracies.append(accuracy_score(y_train, clf.predict(x_train)))
print("Maximum result at:")
max_accuracy_index = np.argmax(test_accuracies)
print(f"Train size at max result: {train_sizes[max_accuracy_index]}")
print(f"Test accuracy at max result: {test_accuracies[max_accuracy_index]}")
print(f"Training error = {1- train_accuracies[max_accuracy_index]}")


plt.plot(train_sizes, train_accuracies,color='blue')
plt.plot(train_sizes,test_accuracies,color = 'green')
plt.scatter(train_sizes[max_accuracy_index], test_accuracies[max_accuracy_index], color = 'red')
plt.xlabel("No: of Training samples")
plt.ylabel("Accuracies")
plt.title("No: of training samples v/s accuracies")
plt.legend(["train","test","max_accuracy"])
plt.show()

#Preforming Guassina mixture models

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
    from scipy.stats import chi2
    from matplotlib.patches import Ellipse
    def eig_sorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order],vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eig_sorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    kwrg = {'facecolor': fc, 'edgecolor': ec, 'alpha': a, 'linewidth': lw}


    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    ax.add_artist(ellip)
    return ellip
def plot_gmm(data,gmm):
    plt.scatter(data[:,0],data[:,1],lw=0)
    plt.xlabel('Principal component-1')
    plt.ylabel('Principal component-2')
    k = gmm.weights_.shape[0]
    for c in range(0,k):
        plot_cov_ellipse(gmm.covariances_[c], gmm.means_[c])
    plt.show()
gmix = GaussianMixture(n_components=2,covariance_type="full")
gmix.fit(x_train_pca)
plot_gmm(x_train_pca,gmix)



