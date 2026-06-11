import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
data=pd.read_csv("bitcoin_dataset.csv")
data=data.drop("Adj Close",axis=1)
data["Date"]=pd.to_datetime(data["Date"])
data.set_index("Date",inplace=True)
corr_matrix=data.corr()
sns.heatmap(corr_matrix,annot=True)
plt.show()
data=data.drop("High",axis=1)
data=data.drop("Low",axis=1)
print(data.head())
def standardize(X):
    C=X.mean()
    return (X-C)/np.std(X,axis=0)
X = data[["Open", "Volume"]]
y = data["Close"]
X = standardize(X)
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)
def predict(X,w,b):
    return np.dot(X,w)+b
def compute_loss(y,y_pred):
    return np.mean(((y-y_pred)**2))
def gradient_de_descente(X,y,w,b,a,N):
    losses=[]
    for i in range(N):
        y_pred=predict(X,w,b)
        dw=(-2/N)*X.T@(y-y_pred)
        db=(-2/N)*np.sum(y-y_pred)
        w=w-a*dw
        b=b-a*db
        losses.append(compute_loss(y,y_pred))
    return w,b,losses
w=np.zeros(2)
b=0
a=0.01
N=1000
w,b,losses=gradient_de_descente(X,y,w,b,a,N)
print("w:", w)
print("b:", b)
y_pred = predict(X, w, b)
print("y:",y_pred)
print("MSE:", compute_loss(y, y_pred))
plt.plot(y, label="Vraies valeurs")
plt.plot(y_pred, label="Prédictions")
plt.title("Bitcoin - Prédiction vs Réalité")
plt.xlabel("Jours")
plt.ylabel("Prix (USD)")
plt.legend()
plt.show()
plt.plot(range(N),losses)
plt.xlabel("n epoch")
plt.ylabel("loss")
plt.show()
def r2_score(y,y_pred) :
    return 1- ((sum((y-y_pred)**2))/sum((y-y.mean())**2))
print("R²:", r2_score(y, y_pred))