import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
df = pd.read_csv("Dry_Bean_Dataset.csv", nrows=10000)
df["Class"] = pd.Categorical(df["Class"], categories=["SEKER", "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SIRA"])
df["Class"] = df["Class"].cat.codes
df.info()
print(df.isnull().sum())
print(df.describe())
pd.crosstab(df["Class"], df["Area"])
print(df.groupby("Class").mean())
iter_vec = []
accuracy_vec = []
results_df = pd.DataFrame(columns=['sample', 'accuracy', 'kernel', 'nu', 'epsilon'])
for sample_num in range(1, 11):
    bestAccuracy = 0
    bestKernel = ""
    bestNu = 0
    bestEpsilon = 0
    iteration = 1000
    kernelList = ['rbf', 'poly', 'linear', 'sigmoid']
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    def fitnessFunction(k, n, e):
        model = svm.NuSVC(kernel=k, nu=n, gamma='auto', cache_size=1000)
        model.fit(X_train, Y_train)
        predicted = model.predict(X_test)
        accuracy = round(np.mean(Y_test == predicted) * 100, 2)
        return accuracy  
    for i in range(1, iteration+1):
        print("Sample:", sample_num, ", Iteration:", i)
        k = np.random.choice(kernelList)
        n = np.random.uniform()
        e = np.random.uniform()
        Accuracy = fitnessFunction(k, n, e)
        if Accuracy > bestAccuracy:
            bestKernel = k
            bestNu = n
            bestEpsilon = e
            bestAccuracy = Accuracy
        bestAccuracy = max(Accuracy, bestAccuracy)
        iter_vec = np.append(iter_vec, i + (sample_num-1)*iteration)
        accuracy_vec = np.append(accuracy_vec, bestAccuracy)
    new_row = pd.DataFrame([[sample_num, bestAccuracy, bestKernel, bestNu, bestEpsilon]], 
                           columns=['sample', 'accuracy', 'kernel', 'nu', 'epsilon'])
    results_df = pd.concat([results_df, new_row], ignore_index=True)
results_df.to_csv("results.csv", index=False)
max_sample = np.argmax(results_df['accuracy'])
max_iter_vec = iter_vec[(iter_vec > (max_sample - 1)*iteration) & (iter_vec <= max_sample*iteration)]
max_accuracy_vec = accuracy_vec[(iter_vec > (max_sample - 1)*iteration) & (iter_vec <= max_sample*iteration)]
plt.plot(max_iter_vec, max_accuracy_vec, '-o')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()