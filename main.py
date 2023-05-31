from flask import Flask, request
from flask_restx import Api, Resource
import os
import json
import pandas as pd
import joblib
import sklearn
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)
api = Api(app, version='1.0', title='Your API', description='API documentation with Swagger')


ns = api.namespace('api', description='Loan API calls')

@ns.route('/loan')
class LoadDefault(Resource):
    def post(self):
        data = request.get_json()
        print(data)  # Retrieve the JSON data from the request
        df2 = pd.json_normalize(data)
        
        #print(df)
        df2['term'] = df2['term'].astype(str).astype(int)
        df2['grade'] = df2['grade'].astype(str).astype(int)
        df2['subgrade'] = df2['subgrade'].astype(str).astype(int)
        df2['verification_status'] = df2['verification_status'].astype(str).astype(int)

        #loaded_model = joblib.load('random_forest_model.joblib')

        #probabilities = loaded_model.predict_proba(df2)

        # Print the predicted probabilities for each sample
        #for i, probs in enumerate(probabilities):
        #    class_0_prob = probs[0]
        #    class_1_prob = probs[1]
        #    string = "Probability to be fully paid: {0:.2f}, Probability to default: {1:.2f}".format(class_0_prob,class_1_prob)

        # Perform further processing with the DataFrame if needed

        # Convert DataFrame to JSON
        #response_data = df.to_json(orient='records')
        string = "Fine"
        # Return JSON response
        return {"message":string}#response_data


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
