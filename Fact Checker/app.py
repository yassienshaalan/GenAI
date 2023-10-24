import streamlit as st
from BertFactChecker import *


def main():
    bert_checker = FactCheckerBERT()
    # Set the title of the app
    st.title("Fact Checker App")

    # Textbox for user input
    evidence = st.text_area("Enter the evidence:")
    claim = st.text_area("Enter the claim:")

    # Button to trigger the fact-checking process
    if st.button("Check"):
        if evidence and claim:
            # Call the fact-checking function
            result = bert_checker.predict(claim,evidence)

            # Display the result
            st.text(f"Is the claim true? {result}")
        else:
            st.warning("Please enter evidence and a claim to check.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
