from flask import Flask, request
from flask_restx import Api, Resource
import os
import json
import pandas as pd
from sklearn.externals import joblib
import sklearn
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)
api = Api(app, version='1.0', title='Your API',
          description='API documentation with Swagger')


ns = api.namespace('api', description='Loan API calls')

loaded_model = joblib.load('random_forest_model.joblib')


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

            probabilities = loaded_model.predict_proba(df2)

            for i, probs in enumerate(probabilities):
                class_0_prob = probs[0]
                class_1_prob = probs[1]
                string = 'Probability to be fully paid: {0:.0f}%. Probability to default: {1:.0f}%'.format(
                    class_0_prob*100, class_1_prob*100)

            return { 'message': string }

        except Exception as e:
            return {'error': str(e)}, 500


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
