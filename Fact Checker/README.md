##Fact Checker APP with BERT
This project provides a fact-checking solution using the BERT model. Specifically, it is designed to verify the truthfulness of a claim based on provided evidence. The BERT model used here has been trained on the fever dataset.

###Table of Contents
Dependencies
Structure
Usage
Training the Model
Using the Model in Streamlit App
Future Work
Contribution
License
Dependencies
transformers
datasets
sklearn
torch
streamlit (for the web app)
Structure
FactCheckerBERT class: Contains methods for initializing the model, preparing the dataset, training the model, evaluating the model, and making predictions.
Streamlit app: A simple web application that allows users to input a claim and its evidence, and then checks the truthfulness of the claim using the trained model.

####Usage
Training the Model

python
Copy code
'''
from BertFactChecker import *

bert_checker = FactCheckerBERT()
bert_checker.get_dataset()
bert_checker.train()
test_accuracy, test_loss = bert_checker.evaluate()
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
'''
Using the Model in Streamlit App
To run the Streamlit app, use:

bash
Copy code
'''
streamlit run app.py
'''
Within the app:

Enter the evidence and the claim you want to verify.
Click on the "Check" button.
The app will display whether the claim is true based on the provided evidence.

####Future Work
Extend the model to support other languages.
Integrate with more datasets for broader fact-checking capabilities.
Improve the user interface of the Streamlit app for a better user experience.
####Contribution
Contributions are welcome! Please submit PRs for any enhancements, bug fixes, or features you might want to add.

####License
This project is licensed under the MIT License.