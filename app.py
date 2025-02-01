from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Load model, scaler, and encoders
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    # Get form data
    features = {}
    features['Gender'] = label_encoders['Gender'].transform([request.form['Gender']])[0]
    features['Married'] = label_encoders['Married'].transform([request.form['Married']])[0]
    features['Dependents'] = label_encoders['Dependents'].transform([request.form['Dependents']])[0]
    features['Education'] = label_encoders['Education'].transform([request.form['Education']])[0]
    features['Self_Employed'] = label_encoders['Self_Employed'].transform([request.form['Self_Employed']])[0]
    features['Property_Area'] = label_encoders['Property_Area'].transform([request.form['Property_Area']])[0]
    features['Credit_History'] = float(request.form['Credit_History'])
    features['ApplicantIncome'] = float(request.form['ApplicantIncome'])
    features['CoapplicantIncome'] = float(request.form['CoapplicantIncome'])
    features['LoanAmount'] = float(request.form['LoanAmount'])
    features['Loan_Amount_Term'] = float(request.form['Loan_Amount_Term'])
    
    # Create input array
    input_data = np.array([[features[col] for col in features]])
    input_data[:, [6, 7, 8, 9, 10]] = scaler.transform(input_data[:, [6, 7, 8, 9, 10]])
    
    # Make prediction
    prediction = model.predict(input_data)
    output = label_encoders['Loan_Status'].inverse_transform(prediction)[0]
    
    # Generate plot
    labels = ['Rejected', 'Approved']
    values = [1 - prediction[0], prediction[0]]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['red', 'green'])
    ax.set_ylabel('Probability')
    plt.title('Loan Approval Prediction')
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return render_template('index.html', pred_text=f'Loan Approved? {output}', plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
