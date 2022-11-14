import numpy as np
import pandas as pd
import json

class Data:
    def getData():
        car_csv = pd.read_csv('./carResaleValue.csv')
        car_make_name = car_csv['make'].unique()
        body_type = car_csv['body_type'].unique()
        car_data = car_csv['car_name'].tolist()
        fuel_type=car_csv['fuel_type'].unique().tolist()
        body_type=car_csv['body_type'].unique()
        body_type=[x for x in body_type if x == x]
        transmission=car_csv['transmission'].unique()
        transmission=[x for x in transmission if x == x]
        cars = []

        for i in car_csv['make']+":"+car_csv['model'].tolist():
            cars.append(str(i).split(":",1))

        car_name = {}
        for i in car_make_name:
            car_name[i] = []

        for car in cars:
            if car[1] not in car_name[car[0]]:
                car_name[car[0]].append(car[1])
        data={}
        data["car_name"] = car_name
        data["fuel_type"]=fuel_type
        data["body_type"]=body_type
        data["transmission"]=transmission
        return data
    def dataLabels():
        df = pd.read_csv('./CarResaleValue.csv')
        datalabels={}
        carname =df.car_name.unique()
        cndict = {}
        for i in range(len(carname)):
            cndict[carname[i]]=i
        datalabels['car_name']=cndict
        fueltype=df.fuel_type.unique()
        fty_dict ={}
        for i in range(len(fueltype)):
            fty_dict[fueltype[i]]=i
        datalabels['fuel_type']=fty_dict
        # pass fty_dict[user_fueltype] to predict
        bodytype=df.body_type.unique()
        bt_dict ={}
        for i in range(len(bodytype)):
            bt_dict[bodytype[i]]=i
        datalabels['body_type']=bt_dict
        # pass bt_dict[user_bodytype] to predict
        trans=df.transmission.unique()
        tr_dict ={}
        for i in range(len(trans)):
            tr_dict[trans[i]]=i
        datalabels['transmission']=tr_dict
        # pass tr_dict[user_transmission] to predict
        make=df.make.unique()
        mk_dict ={}
        for i in range(len(make)):
            mk_dict[make[i]]=i
        datalabels['make']=mk_dict
        # pass mk_dict[user_make] to predict
        model=df.model.unique()
        md_dict ={}
        for i in range(len(model)):
            md_dict[model[i]]=i
        datalabels['model']=md_dict
        # pass md_dict[user_model] to predict
        yn_dict ={"Yes":1,"No":0}
        tf_dict ={"TRUE":1,"FALSE":0}
        datalabels['reserved']=tf_dict
        datalabels['warranty_avail']=tf_dict
        datalabels['paint']=yn_dict
        datalabels['damage']=yn_dict
        return datalabels
print(Data.getData())