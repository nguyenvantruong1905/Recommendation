from flask import Flask,jsonify
from flask_restful import Api, Resource
from class_recommen import Update_Params_And_Recommend
app = Flask(__name__)
api = Api(app)

api.add_resource(Update_Params_And_Recommend, '/recommend')
if __name__ == "__main__":
  app.run(host='0.0.0.0')