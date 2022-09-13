import numpy as np
import pandas as pd
import pickle

class hpp():
    def __init__(self,data):
        self.data=data

    def load_model(self):
        with open(r'artifacts/hpp project.pkl','rb') as file:
            self.model = pickle.load(file)


    def predict(self):

        self.load_model()

        CRIM=float(self.data['CRIM'])
        ZN=float(self.data['ZN'])
        INDUS=float(self.data['INDUS'])
        CHAS=float(self.data['CHAS'])
        NOX=float(self.data['NOX'])
        RM=float(self.data['RM'])
        AGE=float(self.data['AGE'])
        DIS=float(self.data['DIS'])
        RAD=float(self.data['RAD'])
        TAX=float(self.data['TAX'])
        PTRATIO=float(self.data['PTRATIO'])
        B=float(self.data['B'])
        LSTAT=float(self.data['LSTAT'])
        array=np.array([CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT], ndmin=2)
        print(array)

        res = np.around(self.model.predict(array)[0],2)
        print(res)

        return res

if __name__=='__main__':
    data={'CRIM':0.00832,
    'ZN':16.00000,
    'INDUS':3.31000,
    'CHAS':0.00000,
    'NOX':0.23800,
    'RM':3.57500,
    'AGE':57.20000,
    'DIS':6.09000,
    'RAD':1.00000,
    'TAX':305.00000,
    'PTRATIO':19.30000,
    'B':350.90000,
    'LSTAT':2.98000}
    hpp_obj=hpp(data)

    hpp_obj.predict()

    