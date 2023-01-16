# Databricks notebook source
# MAGIC %pip install keras tensorflow

# COMMAND ----------

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

# COMMAND ----------

from pyspark.sql.functions import rand
from pyspark.mllib.evaluation import MulticlassMetrics

# COMMAND ----------

from pyspark.sql.functions import rank
from pyspark.sql import Window

# COMMAND ----------

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers, regularizers
from tensorflow.keras.optimizers import Adam

# COMMAND ----------

conf = SparkConf().setAppName('Stock Pred Pipeline').setMaster('local[*]')
sc = SparkContext.getOrCreate(conf=conf)
sql_context = SQLContext(sc)

# COMMAND ----------

df = sql_context.read.csv('/FileStore/tables/data.csv',
                    header=True,
                    inferSchema=True)

# COMMAND ----------

df = df.withColumn("rank", rank().over(Window.partitionBy().orderBy("date")))

# COMMAND ----------

train_df = df.where("rank <= 1625").drop("rank","Open","High","Low","Volume","OpenInt")

# COMMAND ----------

test_df = df.where("rank > 1575").drop("rank","Open","High","Low","Volume","OpenInt")

# COMMAND ----------

from pyspark.sql.functions import col, stddev_samp

# COMMAND ----------

maxVal_train = train_df.agg({'Close':'max'}).collect()[0][0]
minVal_train = train_df.agg({'Close':'min'}).collect()[0][0]

# COMMAND ----------

train_dff = train_df.withColumn("ScaledClose",
  (col("Close") - minVal_train)/(maxVal_train - minVal_train))

# COMMAND ----------

maxVal_test = test_df.agg({'Close':'max'}).collect()[0][0]
minVal_test = test_df.agg({'Close':'min'}).collect()[0][0]

# COMMAND ----------

test_dff = test_df.withColumn("ScaledClose",
  (col("Close") - minVal_train)/(maxVal_train - minVal_train))

# COMMAND ----------

price_col_train = train_dff.select("ScaledClose").rdd.map(lambda x: x[0]).collect()
price_col_test = test_dff.select("ScaledClose").rdd.map(lambda x: x[0]).collect()

# COMMAND ----------

import numpy as np

# COMMAND ----------

price_col_arr_train = np.array(price_col_train)
price_col_arr_test = np.array(price_col_test)

# COMMAND ----------

price_col_arr_train = price_col_arr_train.reshape(-1,1)
price_col_arr_test = price_col_arr_test.reshape(-1,1)

# COMMAND ----------

def generateDataPatterns(scaled_train_data,history):   
    x_history = []  
    y_price_cur = []
    train_len = scaled_train_data.shape[0]
    day = history
    while day<train_len:
        y_cur_price = scaled_train_data[day,0]
        cur_range = scaled_train_data[day-history:day,0]
        x_history.append(cur_range)
        y_price_cur.append(y_cur_price)
        day+=1
    
    x_history = np.array(x_history)
    y_price_cur = np.array(y_price_cur)
    
    x_history = x_history.reshape(x_history.shape[0],x_history.shape[1],1)
    return x_history,y_price_cur


# COMMAND ----------

pattern_train_data = generateDataPatterns(price_col_arr_train,50)
trainX = pattern_train_data[0]
trainY = pattern_train_data[1]
trainX.shape

# COMMAND ----------

trainY.shape

# COMMAND ----------

pattern_test_data = generateDataPatterns(price_col_arr_test,50)
testX = pattern_test_data[0]
testY = pattern_test_data[1]
testX.shape

# COMMAND ----------

testY = testY.reshape(-1,1)

# COMMAND ----------

testY.shape

# COMMAND ----------

from tensorflow.keras.layers import SimpleRNN, LSTM

# COMMAND ----------

class RNNStockModel():
 
    loss_function ='mean_squared_error'
    batch_size=32
    num_neu = 50
    model = Sequential()
    def __init__(self,trainX,trainY,epoch):
        self.trainX = trainX
        self.trainY = trainY
        self.epoch = epoch
    
    def buildModel(self):
        RNNStockModel.model = Sequential()
        RNNStockModel.model.add(SimpleRNN(RNNStockModel.num_neu,
                                            activation='tanh',
                                            return_sequences = True,
                                            input_shape = (self.trainX.shape[1],1)))
        
        RNNStockModel.model.add(Dropout(0.2))
        
        RNNStockModel.model.add(SimpleRNN(RNNStockModel.num_neu,
                                            activation='tanh',
                                            return_sequences = True))
        RNNStockModel.model.add(Dropout(0.2))
        
        RNNStockModel.model.add(SimpleRNN(RNNStockModel.num_neu,
                                            activation='tanh',
                                            return_sequences = True))
        RNNStockModel.model.add(Dropout(0.2))
        
        RNNStockModel.model.add(SimpleRNN(RNNStockModel.num_neu,
                                            activation='tanh',
                                            return_sequences = False))
        
        RNNStockModel.model.add(Dropout(0.2))
        
        RNNStockModel.model.add(Dense(units=RNNStockModel.num_neu,
                                        activation='tanh'))
        
        RNNStockModel.model.add(Dense(units=1))
        return RNNStockModel.model.summary()
    
    def model_fit(self):
        prev = RNNStockModel.model.fit(self.trainX,self.trainY,
                                        epochs=self.epoch,batch_size=RNNStockModel.batch_size,validation_split=0.2,
                                       )
        return prev
    
    def NeuronsUpdate(self,neu):
        RNNStockModel.num_neu = neu
 
    def EpochUpdate(self,epoch):
        self.epoch = epoch
    
    def BatchSizeUpdate(self,cur_batch_size):
        RNNStockModel.batch_size = cur_batch_size
    
    def model_compile(self):
        RNNStockModel.model.compile(optimizer = Adam(),
                                    loss = RNNStockModel.loss_function)
        return RNNStockModel.model.summary()
    
    def evaluateModel(self, x=None, y=None):
        if x == None:
            x = self.trainX
        if y == None:
            y = self.trainY
        scores = RNNStockModel.model.evaluate(x, y, verbose = 0)
        return scores

# COMMAND ----------

class LSTMModel(RNNStockModel):
    RNNStockModel.model = Sequential()
    def __init__(self,trainX,trainY,epoch):
        super().__init__(trainX,trainY,epoch)
    
    def buildModel(self,dense=1):
        RNNStockModel.model = Sequential()
        RNNStockModel.model.add(LSTM(
                                 RNNStockModel.num_neu,
                                 input_shape=(None,1)))
        
        RNNStockModel.model.add(Dense(units=1))
        return RNNStockModel.model.summary()

# COMMAND ----------

import matplotlib.pyplot as plt
%matplotlib inline

# COMMAND ----------

def plot_graph(original_val,predicted_val):
    plt.figure(figsize=(10,5), dpi=80, facecolor='w', edgecolor='k')
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.plot(original_val,color="Red",label="Original Stock Price")
    plt.plot(predicted_val,color="Blue",label="Predicted Stock Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# COMMAND ----------

for epoch in [16]:
    for batch_size in [70]:
        for num_neurons in [80]:
            FinalModel = LSTMModel(trainX,trainY,epoch=epoch)
            FinalModel.NeuronsUpdate(num_neurons)
            FinalModel.BatchSizeUpdate(batch_size)
            FinalModel.buildModel()
            FinalModel.model_compile()
            prev = FinalModel.model_fit()
            
            plt.plot(prev.history['loss'])
            plt.plot(prev.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()

            predicted_val = FinalModel.model.predict(testX)
            predicted_val2 = sc.parallelize(predicted_val).map(lambda x: x[0]*(maxVal_train- minVal_train)+minVal_train).collect()           
            original_val = sc.parallelize(testY).map(lambda x: x[0]*(maxVal_train- minVal_train)+minVal_train).collect()
            diff = [(a-b)*(a-b) for a,b in zip(predicted_val2, original_val)]
            
            import math
            rmse = math.sqrt(sum(diff)/len(diff))
            print("For epoch {}, neurons {} and batch_size {}".format(epoch,num_neurons,batch_size))
            print("RMSE : {}".format(rmse))
            plot_graph(original_val,predicted_val2)

# COMMAND ----------

#Epochs
rmse=[]
for epoch in [2,4,8,16,32,64]:
    FinalModel = LSTMModel(trainX,trainY,epoch)
    FinalModel.NeuronsUpdate(80)
    FinalModel.BatchSizeUpdate(70)
    FinalModel.buildModel()
    FinalModel.model_compile()
    prev = FinalModel.model_fit()
    predicted_val = FinalModel.model.predict(testX)
    predicted_val2 = sc.parallelize(predicted_val).map(lambda x: x[0]*(maxVal_train- minVal_train)+minVal_train).collect()
    original_val = sc.parallelize(testY).map(lambda x: x[0]*(maxVal_train- minVal_train)+minVal_train).collect()
    diff = [(a-b)*(a-b) for a,b in zip(predicted_val2, original_val)]
    import math
    rmse.append(math.sqrt(sum(diff)/len(diff)))
    print("For epoch {}, neurons {} and batch_size {}".format(epoch,num_neurons,batch_size))

# COMMAND ----------

epochs = [2,4,8,16,32,64]
print(rmse)
plt.plot()
plt.plot(epochs,rmse)
plt.title('rmse vs epoch')
plt.ylabel('rmse')
plt.xlabel('epochs')
plt.show()

# COMMAND ----------

#Batch_Size
rmse=[]
for batch_size in [10,20,30,40,50,60,70,80,90,100]:
    FinalModel = LSTMModel(trainX,trainY,epoch=16)
    FinalModel.NeuronsUpdate(80)
    FinalModel.BatchSizeUpdate(batch_size)
    FinalModel.buildModel()
    FinalModel.model_compile()
    prev = FinalModel.model_fit() 
    predicted_val = FinalModel.model.predict(testX)
    predicted_val2 = sc.parallelize(predicted_val).map(lambda x: x[0]*(maxVal_train- minVal_train)+minVal_train).collect()           
    original_val = sc.parallelize(testY).map(lambda x: x[0]*(maxVal_train- minVal_train)+minVal_train).collect()
    diff = [(a-b)*(a-b) for a,b in zip(predicted_val2, original_val)]
    import math
    rmse.append(math.sqrt(sum(diff)/len(diff)))
    print("For epoch {}, neurons {} and batch_size {}".format(epoch,num_neurons,batch_size))

# COMMAND ----------

batch_size = [10,20,30,40,50,60,70,80,90,100]
print(rmse)
plt.plot()
plt.plot(batch_size,rmse)
plt.title('rmse vs batch_size')
plt.ylabel('rmse')
plt.xlabel('batch_size')
plt.show()

# COMMAND ----------

#Neurons
rmse=[]
for num_neurons in [40,80,120,160,200,240,280,320]:
    FinalModel = LSTMModel(trainX,trainY,epoch = 16)
    FinalModel.NeuronsUpdate(num_neurons)
    FinalModel.BatchSizeUpdate(70)
    FinalModel.buildModel()
    FinalModel.model_compile()
    prev = FinalModel.model_fit()
    predicted_val = FinalModel.model.predict(testX)
    predicted_val2 = sc.parallelize(predicted_val).map(lambda x: x[0]*(maxVal_train- minVal_train)+minVal_train).collect()           
    original_val = sc.parallelize(testY).map(lambda x: x[0]*(maxVal_train- minVal_train)+minVal_train).collect()
    diff = [(a-b)*(a-b) for a,b in zip(predicted_val2, original_val)]
    import math
    rmse.append(math.sqrt(sum(diff)/len(diff)))
    print("For epoch {}, neurons {} and batch_size {}".format(epoch,num_neurons,batch_size))

# COMMAND ----------

neurons = [40,80,120,160,200,240,280,320]
print(rmse)
plt.plot()
plt.plot(neurons,rmse)
plt.title('rmse vs num_neurons')
plt.ylabel('rmse')
plt.xlabel('num_neurons')
plt.show()
