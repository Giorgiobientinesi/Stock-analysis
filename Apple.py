import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers
from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

df = pd.read_csv('Apple.csv', error_bad_lines=False)

print(df)
lst = []
i = 0
while i < 2359:
    lst.append(i)
    i +=5
lst1= []
for k in range (1,2359):
    lst1.append(k)

for el in lst1:
    for i in lst:
        if el == i:
            lst1.remove(el)


for el in lst1:
    df = df.drop(index=el , errors='raise')






a = df["Close/Last"].tolist()
b = []

for el in a:
    b.append(float(el[1:]))

df = pd.DataFrame(b)


df.columns = ['Close']



print(df)



df["Close"] = df["Close"].values[::-1]
df = df.reset_index()
del df['index']


sns.set(style="darkgrid", font_scale= 1.5)
plt.figure(figsize=(12,6))
sns.lineplot(x=df.index, y="Close", data=df).set_title("Apple")
plt.show()


data = df.iloc[:, 0]
hist = []
target = []
length = 50
for i in range(len(data)-length):
    x = data[i:i+length]
    y = data[i+length]
    hist.append(x)
    target.append(y)

hist = np.array(hist)
target = np.array(target)
target = target.reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
hist_scaled = sc.fit_transform(hist)
target_scaled = sc.fit_transform(target)

hist_scaled = hist_scaled.reshape((len(hist_scaled), length, 1))
print(hist_scaled.shape)


X_train = hist_scaled[:2200,:,:]
X_test = hist_scaled[2200:,:,:]
y_train = target_scaled[:2200,:]
y_test = target_scaled[2200:,:]


print(X_train)
print(X_test)
#print(y_train)
#print(y_test)  #gli ultimi 14 giorni



model = Sequential()

model.add(layers.SimpleRNN(units=50, return_sequences=True, input_shape=(50,1),kernel_initializer='glorot_uniform',
                    bias_initializer='glorot_uniform',))

model.add(layers.SimpleRNN(units=50, return_sequences=True,kernel_initializer='glorot_uniform',
                    bias_initializer='glorot_uniform',))

model.add(Dropout(0.2))
model.add(layers.SimpleRNN(units=50))
model.add(Dropout(0.2))

model.add(Dense(32,activation="relu"))

model.add(layers.Dense(units=1))


model.compile(loss= "mean_squared_error", optimizer= "adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=100, batch_size=32)

pred = model.predict(X_test)



plt.figure(figsize=(12,8))
plt.plot(y_test, color='blue', label='Real')
plt.plot(pred, color='red', label='Prediction')
plt.title('Euro Prediction')
plt.legend()
plt.show()

pred_transformed = sc.inverse_transform(pred)
print(pred_transformed)

y_test_trasnformed = sc.inverse_transform(y_test)
print(y_test_trasnformed)



list_pred = []

for el in pred_transformed:
    for i in el:
        list_pred.append(i)

print(list_pred)

list = []

for j in y_test_trasnformed:
    for k in j:
        list.append(k)

i = 0
counter = 0
print(len(list_pred))


print((list_pred))
print(list)
print(len(list))

while i <= 107:
    if (list_pred[i] > list_pred[i+1] and list[i] > list[i+1]) or (list_pred[i] < list_pred[i+1] and list[i] < list[i+1]) or (list_pred[i] == list_pred[i+1] and list[i] == list[i+4]):
        counter += 1
    i += 1

print(counter)
