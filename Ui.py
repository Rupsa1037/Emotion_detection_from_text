import streamlit as st 
import joblib

#loading and vectorizing the model
# model = joblib.load("rf_emotion_model.pkl")
model = joblib.load("Models/rf_emotion_model.pkl")
# vectorizer = joblib.load("rf_vectorizer.pkl")
vectorizer = joblib.load("Models/rf_vectorizer.pkl")


try:
    le = joblib.load("label_encoder.pkl")
    use_label_encoder = True
except:
    use_label_encoder = False

#function for precicting emotion
def predict_emotion(text):
    #vector = vectorizer.transform([text])
    vector=vectorizer.encode(text).reshape(1,-1)
    prediction = model.predict(vector)
    if use_label_encoder:
        return le.inverse_transform(prediction)[0]
    else:
        return prediction[0]

#layout for app
st.title(": Emotion Detection from Text :")
st.subheader(": Enter a sentence to detect the emotion :")

user_input = st.text_area("Your Text", height=150)

if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        result = predict_emotion(user_input)
        st.success(f"--> Predicted Emotion: **{result}**")
