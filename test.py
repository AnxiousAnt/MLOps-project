import pickle


model = pickle.load(open('model.pkl', 'rb'))
print(type(model))  # Check the type of the loaded model