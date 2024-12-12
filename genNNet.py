print("0.生成RESNET神经网络模型")
################################################
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import  plot_model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn import tree

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import copy
from imblearn.over_sampling import RandomOverSampler

import pickle  


import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import  plot_model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn import tree

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import copy
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pickle 
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes

from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

import sys
import argparse

from sklearn.model_selection import train_test_split

########################################################################################################################
###简单模型3，resnet_like
def local_model(num_labels, dropout_rate, relu_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(relu_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_labels, activation='sigmoid'))
    return model

def global_model(dropout_rate, relu_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(relu_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    return model



def sigmoid_model(label_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(label_size, activation='sigmoid',name="global"))
    return model

def softmax_model(label_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(label_size, activation='softmax',name="global"))
    return model

def getKerasResnetRVL(x,enc,saveName):
    print(saveName)
    model_name = saveName 
    model = keras.models.load_model(model_name)
    y= model.predict([x], batch_size=2560)
    nSamples = y.shape[0]
    ###需要将预测出的值，转换01整数,并转为数字式
    for i in range(y.shape[0]):
        tmp = y[i]
        index=  np.argmax(tmp)
        y[i] = [0]*y.shape[1]
        y[i,index]=1

    ###  
    y= enc.inverse_transform(y)
    y= y.reshape(-1,nSamples)[0]
    
    
    return y


######################################################################################################################## 
def setNoTrainedLayers(model,noTrainingLayers):
 
    #只训练最后一层，#
    #last_layer_idx = len(model.layers) - 1
    #noTrainingLayers = range(last_layer_idx)
    for i in noTrainingLayers:
        model.layers[i].trainable = False

    return model


############################################################################
####HMCM-F ,层次模型，发现hmcn-f训练效果很差，所以采用分离式
###每一层的识别模型都是4层模型
##分层重新训练，加入特征SMV1,SMV2
def sepHier1_SUMO(x,yOneHot,num_labels,saveName,levelIndex,numLayers,numEpochs = 10,srelu_size = 256,dropout_rate = 0.05):
    
    str1="layIndex-"+str(levelIndex)
    
    nSamples,features_size = x.shape
    relu_size = 256
    dropout_rate = 0.05
    global_models = []
    
    label_size = num_labels
    featuresInput = layers.Input(shape=(features_size,))
    features = layers.BatchNormalization()(featuresInput)
    #features=featuresInput
    for i in range(numLayers):
        if i == 0:
            global_models.append(global_model(dropout_rate, relu_size)(features))
        else:
            global_models.append(global_model(dropout_rate, relu_size)(layers.concatenate([global_models[i-1], features])))
    
    p_glob = softmax_model(label_size)(global_models[-1])
    build_model = tf.keras.Model(inputs=[featuresInput], outputs=[p_glob])

    
    build_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    if 0:
        build_model = keras.models.load_model(saveName)
    if 1:#用于画图
        #build_model.fit([x],[yOneHot],epochs=1, batch_size=10000*1)
        #build_model.summary()
        build_model.fit(x,yOneHot,epochs=1, batch_size=10000*1)
        plot_model(build_model, to_file=str1+".jpg", show_shapes=True)
    
  
    build_model.fit(x,yOneHot,epochs=numEpochs,batch_size=160000*1)#GPU用这个
    build_model.save(saveName)
    return build_model


'''
生成简单模型1:
1. levelIndex = 0 永远等于=0，不做层次

'''
def simpleNN1(x,yOneHot,num_labels,saveName,levelIndex = 0,numLayers,numEpochs = 10,srelu_size = 256,dropout_rate = 0.05):
    
    levelIndex = 0
    str1="layIndex-"+str(levelIndex)
    
    nSamples,features_size = x.shape
    relu_size = 256
    dropout_rate = 0.05
    global_models = []
    
    label_size = num_labels
    featuresInput = layers.Input(shape=(features_size,))
    features = layers.BatchNormalization()(featuresInput)
    #features=featuresInput
    for i in range(numLayers):
        if i == 0:
            global_models.append(global_model(dropout_rate, relu_size)(features))
        else:
            global_models.append(global_model(dropout_rate, relu_size)(layers.concatenate([global_models[i-1], features])))
    
    p_glob = softmax_model(label_size)(global_models[-1])
    build_model = tf.keras.Model(inputs=[featuresInput], outputs=[p_glob])

    
    build_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    if 0:
        build_model = keras.models.load_model(saveName)
    if 1:#用于画图
        #build_model.fit([x],[yOneHot],epochs=1, batch_size=10000*1)
        #build_model.summary()
        build_model.fit(x,yOneHot,epochs=1, batch_size=10000*1)
        plot_model(build_model, to_file=str1+".jpg", show_shapes=True)
    
  
    build_model.fit(x,yOneHot,epochs=numEpochs,batch_size=160000*1)#GPU用这个
    build_model.save(saveName)
    return build_model




########################################################################################################################
########################################################################################################################
########################################################################################################################

def main():
    #python3 genNNet.py --numEpochs 10 --trainOrNot 1 --testOrNot 1 --sampleMethond 1
    parser = argparse.ArgumentParser(description="step0")
    parser.add_argument('-np','--numEpochs', default=1000, type=int,help='分离式模型每层训练次数')
    parser.add_argument('-trn','--trainOrNot', default=1,type=int,help='训练吗?')
    parser.add_argument('-ten','--testOrNot', default=1,type=int,help='测试吗?')
    parser.add_argument('-tr','--testRatio', default = 0.9,type=float,help='测试集比例')
 
    args = parser.parse_args()
    numEpochs = args.numEpochs
    trainOrNot =  args.trainOrNot
    testRatio =  args.testRatio
    testOrNot = args.testOrNot
  
    hierarchy = 0
   

    fs1 = open('simpleNN1_printlog.txt', 'w+')
    sys.stdout = fs1  # 将输出重定向到文件


    ##############################################################################
    print("0.主程序开始, )
    np.random.seed(42)
    tf.random.set_seed(42)





    if trainOrNot == 1:# 训练多级模型
        print("训练分离式多级模型")

        #准备字典，用于保存训练后的数据"
        xFloors=  dict()
        yFloors =  dict()
        xTestFloors =dict()
        yTestFloors = dict()
        modSaveNameFloors =dict()
        encLevels= dict()
        yKerasFloors = dict()
        x=x0
        y=y0
        x=x.astype(np.float32)#GPU 加这个
        y=y.astype(np.int64)#GPU 加这个
        print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )
        print(y)

     


        x_train, x_test, y_train, y_test = train_test_split(x, yH1, test_size=testRatio, random_state=0)

        nSamples,nFeatures =  x_train.shape


        #numEpochs =10 #1500/60/60*5 = 2hour #17000正确率过高

        numLayers = 4
        enc = OneHotEncoder()
        nSamples,nFeatures =  x_train.shape
        y_train= np.array(y_train)
        print("y_train.shape:",y_train)

        y_train= y_train.reshape(nSamples,-1)
        enc.fit(y_train)

        yOneHot=enc.transform(y_train).toarray()
        print(enc.categories_,enc.get_feature_names())
        print(yOneHot[:1])


        num_labels = y_train.unique().size
        print("num_labels:", num_labels)
        
        saveName = "./trainedModes/simpleNN1_numlayers%3d_time%s.h5" %(numLayers,datatime.now())
        levelIndex = 0
        build_model = simpleNN1(x_train,yOneHot,num_labels,saveName,levelIndex,numLayers,numEpochs)

        '''
        保存为pickle文件,用于后期的数据分析
        '''
        #fpk=open('samples1.pkf','wb+')  
        #pickle.dump([xFloors,yFloors,modSaveNameFloors,encLevels,xTestFloors, yTestFloors],fpk)  
        #fpk.close() 
          
        pkfSaveName ="./trainedModes/simpleNN1_numlayers%3d.pkf" %(numLayers)
        fpk=open(pkfSaveName,'wb+')  
        pickle.dump( [x_train, x_test, y_train, y_test,enc,modSaveName,build_model],fpk)  
        fpk.close() 

    ################################################################  
    ################################################################
    #####用现有训练模型进行预测

    if testOrNot == 1:
        
        pkfSaveName ="./trainedModes/simpleNN1_numlayers%3d.pkf" %(numLayers)            
        fpk=open(pkfSaveName,'rb') 

        [x_train, x_test, y_train, y_test,enc,modSaveName,build_model]=pickle.load(fpk)  
        fpk.close()  


       
        yPredict=getKerasResnetRVL(x_test,enc,modSaveName)
        mat1num = confusion_matrix(y_test,yPredict)
        print(mat1num)
        mat2acc = confusion_matrix(yCurLayer1,yPredict,normalize='pred')  
        print(np.around(mat2acc , decimals=3))
        yKerasFloors[str(i)] =  yPredict

        df = pd.DataFrame(np.around(mat2acc , decimals=3))
        fs = "test_mat2acc.csv"
        df.to_csv(fs,index= False, header= False)
       

 
      
if __name__=="__main__":
    main()