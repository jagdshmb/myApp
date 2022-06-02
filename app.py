from flask import Flask, request, jsonify, url_for, redirect, render_template
import pickle
import numpy as np
model=pickle.load(open('artifacts/model2.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict_perovskites',methods=['POST','GET'])
def perovskitesPrediction():
  formationEnergy =  float(request.form['formationEnergy'])
  structuralDensity = float(request.form['structuralDensity'])
  octahedralFactor = float(request.form['octahedralFactor'])
  tolerenceFactor = float(request.form['tolerenceFactor'])
  activationEnergy = float(request.form['activationEnergy'])
  specificHeat = float(request.form['specificHeat'])
  heatFusion = float(request.form['heatFusion'])
  vanDerWaalsRadius = float(request.form['vanDerWaalsRadius'])
  meanAtomicRadius = float(request.form['meanAtomicRadius'])
  averageIonicRadius = float(request.form['averageIonicRadius'])
  spBonding = float(request.form['spBonding'])
  electronegativity = float(request.form['electronegativity'])
  final = [formationEnergy, structuralDensity, octahedralFactor, tolerenceFactor, activationEnergy, specificHeat, heatFusion, vanDerWaalsRadius, meanAtomicRadius, averageIonicRadius, spBonding, electronegativity]
  print(final)
  ehull, eg = model.predict([final])[0]
  response = jsonify({
      'estimated_ehull': round(ehull, 2),
      'estimated_eg': round(eg, 2)
  })
  response.headers.add('Access-Control-Allow-Origin', '*')
  return response

if __name__ == "__main__":
    print("Starting Python Flask Server...")
    app.run()
