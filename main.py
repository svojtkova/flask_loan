from flask import Flask, request
from flask_restx import Api, Resource
import os
import json
import numpy as np
import pandas as pd
import pickle
import joblib
from flask_cors import CORS
import zipfile
import os

app = Flask(__name__)
cors = CORS(app)
api = Api(app, version='1.0', title='Your API',
          description='API documentation with Swagger')


ns = api.namespace('api', description='Loan API calls')


zip_path = os.path.join(os.getcwd(), "random_forest_model.zip")

with zipfile.ZipFile(zip_path, 'r') as zip_file:
    zip_file.extractall(os.getcwd())

fname =  os.path.join(os.getcwd(),'random_forest_model.pkl')
loaded_model =  pickle.load(open(fname,'rb'))


@ns.route('/loan')
class LoadDefault(Resource):
    def post(self):
        try:
            string = ''
            data = request.get_json()
            df2 = pd.json_normalize(data)

            df2['term'] = df2['term'].astype(str).astype(int)
            df2['grade'] = df2['grade'].astype(str).astype(int)
            df2['subgrade'] = df2['subgrade'].astype(str).astype(int)
            df2['verification_status'] = df2['verification_status'].astype(
                str).astype(int)
            
            df2 = np.array(df2).astype(np.float32)
            

            probabilities = loaded_model.predict_proba(df2)

            for i, probs in enumerate(probabilities):
                class_0_prob = probs[0]
                class_1_prob = probs[1]
                string = 'Probability to be fully paid: {0:.0f}%. Probability to default: {1:.0f}%'.format(
                    class_0_prob*100, class_1_prob*100)

            return { 'message': string }

        except Exception as e:
            print(str(e))
            return {'error': str(e)}, 500


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
