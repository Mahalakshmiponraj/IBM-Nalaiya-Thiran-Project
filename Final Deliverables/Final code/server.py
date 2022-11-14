import pickle
from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
from json import JSONEncoder
import numpy
import dataParsing
import time

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load the model
model = pickle.load(open("./model.pkl",'rb'))

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

#• Members API Route
@app. route("/members")
def members ():
    # car_name(48),yr_mfr(2017.0),fuel_type(2),kms_run(17406),body_type(0),transmission(2),make(7),model(71),total_owners(1),orginal_price(707825.0),reserved(0),warranty_avail(0),paint(1),damage(0),no_of_service(2),bs(1)
    # ['car_name', 'yr_mfr', 'fuel_type', 'kms_run', 'body_type', 'transmission', 'make', 'model', 'total_owners', 'warranty_avail', 'paint', 'damage', 'no_of_service', 'bs']
    result=model.predict([[48,2017.0,2,17406,0,2,7,71,1,0,1,0,2,1]])
    data={"carvalue": result }
    encodedNumpyData = json.dumps(data, cls=NumpyArrayEncoder)
    return encodedNumpyData

@app.route("/home")
@cross_origin()
def getdata():
    data = dataParsing.Data.getData()
    return data

@app.route("/data", methods=['POST','GET'])
def resposne():
    # print(request.data['car_name'])
    # datas = request.get_json()
    datas = json.loads(request.data)
    mld=dataParsing.Data.dataLabels()
    car_make=mld['make'][datas["car_name"]]
    car_model=mld['model'][datas["car_model"]]
    car_name=mld['car_name'][datas["car_name"]+" "+datas["car_model"]]
    man_year=int(datas['man_year'])
    fuel_type=mld['fuel_type'][datas["fuel_type"]]
    kms_driven=int(datas['kms_driven'])
    body_type=mld['body_type'][datas["body_type"]]
    transmission=mld['transmission'][datas["transmission"]]
    no_of_owners=int(datas['no_of_owners'])
    warranty_avail=1 if datas['warranty']=="YES" else 0
    paint=1 if datas['paint']=="YES" else 0
    damage=1 if datas['damage']=="YES" else 0
    no_of_service=int(datas['no_of_service'])
    bs=int(datas['bs'])
    original_price=int(datas['original_price'])
    print(paint, 'asdsrgdvdsfs',  damage)
    result=model.predict([[car_name,man_year,fuel_type,kms_driven,body_type,transmission,car_make,car_model,no_of_owners,original_price,warranty_avail,paint,damage,no_of_service,bs]])
    data={"carvalue": result }
    encodedNumpyData = json.dumps(data, cls=NumpyArrayEncoder)
    time.sleep(3)
    return encodedNumpyData
if __name__ == "__main__":
    app.run(debug=True)