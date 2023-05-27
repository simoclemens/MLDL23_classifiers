import pandas as pd
import pickle

'''
type = ['train','test']
for t in type:
    myDict = pd.read_pickle('saved_features/dense_5_D1_'+t+'.pkl')

    for elem in myDict['features']:
        elem['features_EMG'] = elem['features_RGB']

    file = open('saved_features/features_'+t+'.pkl','wb')
    pickle.dump(myDict,file)
    print('DONE')
'''
myDict = pd.read_pickle('saved_features_TESTING/features_D1_test.pkl')
for elem in myDict['features']:
    if elem['ver'].shape[0] != 5:
        print(elem['uid'])

    if elem['features_EMG'].shape[0] != 5:
        print(elem['uid'])

