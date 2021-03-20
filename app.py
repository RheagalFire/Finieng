from flask import Flask,request,jsonify,render_template
from model import Translate

app=Flask(__name__)

@app.route('/')
def home():
    return """Hello"""
@app.route('/predict',methods=['POST'])
def predict():
    input_text=request.form['input_text']
    output_text=Translate(input_text)
    return output_text

if __name__ == '__main__':
    app.run(debug=True)


