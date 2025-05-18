# Emotion Detection From Text :
**Introduction :** This project is focused on detecting emotions from text using machine learning techniques. 
It takes input sentences and classifies them into predefined emotion categories such as happy, sad, angry, etc.
## Steps -
- Load and clean the dataset
- Preprocess text using neattext 
- Emotion label distribution analysis using visualizations
- Feature extraction using Sentence Transformer
- Oversampling using SMOTE to balance classes
- Split data into training and test sets (80-20)
- Train models: Logistic Regression, SVM, Random Forest
- Evaluate using accuracy, confusion matrix, classification report
- Saving the trained model and vectorizer using joblib so it can be reused later
- Making predictions using both the Google Colab and the Streamlit web app
## Data Preprocessing -
- Dropped missing values with dropna()
- Cleaned text using:
  - nfx.remove_userhandles
  - nfx.remove_stopwords
  - nfx.remove_punctuations
- Transformed cleaned text into numerical format using SentenceTransformer (all-MiniLM-L6-v2)
- Balanced the dataset using SMOTE
- ## Data Splitting -
- The new balanced dataset was split into training and test datasets where 20% of the data will be used formodel testing and 80% for training the model.
## Model Training
- Three models trained:
  - Logistic Regression (liblinear, max_iter=3000)
  - Support Vector Machine (linear kernel)
  - Random Forest Classifier (best performing)
- Accuracy and classification reports printed for each
- Confusion matrix plotted using seaborn heatmap
- Once training is completed, the trained model is saved and a function is defined with preprocessing steps to predict the emotion of sample examples.
- Afterward, the previously saved model and vectorizer are loaded so that they can be reused later without retraining, especially during deployment.
## How to Run the Streamlit Application -
- Create a new folder on your desktop (or any preferred location) and save all the required .py files and the .pkl model file inside this folder.
- Open the .py files(Ui.py in my case) using IDLE or any Python IDE.
- Run the script in IDLE by navigating to the menu and selecting Run → Run Module to verify there are no immediate errors.
- Right-click on the folder where your files are saved and select Open Terminal.
- In the terminal, run the following command to start the Streamlit application: streamlit run Ui.py
- After running the command, your default web browser will open automatically with the Streamlit app running locally.
  # Streamlit App Demo - ![Screenshot 2025-05-16 224518](https://github.com/user-attachments/assets/371282cd-5017-4c81-b124-9d498b9de97f)
## Future Scope
- Advanced models, such as deep learning, can be applied to increase accuracy.
- Collecting more data, especially for the less accurately predicted emotions.
- Trying different feature extraction techniques.

-Extend to multi-label emotion detection
