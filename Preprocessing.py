import numpy as np

# load data

print("Loading datasets")
X_valid_raw = np.load("data/validx.pkl")
X_train = np.load("data/trainx.pkl")
X_test = np.load("data/testx.pkl")
y_valid = np.load("data/validy.pkl")
y_train = np.load("data/trainy.pkl")
y_test = np.load("data/testy.pkl")


#print(X_valid[0])
#print(X_valid[0][0])
#print(y_valid)

def processDataset(dataset):

    newDataset = []
    for j, x in enumerate(dataset):

        newX = []

        for k, channel in enumerate(x):
            newChannel = []

            # Replace "NA" with NaN
            for i, element in enumerate(channel):

                foo = element.decode('ASCII')
                newChannel.append(foo)

                if newChannel[i] == "NA":
                    newChannel[i] = np.nan

                newChannel[i] = float(newChannel[i])

            newX.append(newChannel)

        newDataset.append(np.array(newX))

    return newDataset


X_valid = processDataset(X_valid_raw)
X_test = processDataset(X_test)
X_train = processDataset(X_train)

print("\nExample data:\n")
print(X_valid[0])
print(X_valid[0][0])

np.save("data/validx_final.npy", X_valid)
np.save("data/trainx_final.npy", X_train)
np.save("data/testx_final.npy", X_test)
np.save("data/validy_final.npy", y_valid)
np.save("data/trainy_final.npy", y_train)
np.save("data/testy_final.npy", y_test)
