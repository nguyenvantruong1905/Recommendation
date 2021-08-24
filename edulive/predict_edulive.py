from flask import Flask,jsonify
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

api.add_resource(Recommend, '/')

if __name__ == "__main__":
  app.run(host='0.0.0.0')