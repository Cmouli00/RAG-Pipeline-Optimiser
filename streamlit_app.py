import streamlit as st
import requests
import json

st.title("RAG Pipeline Optimizer")
uploaded_file = st.file_uploader("Upload Knowledge Base (PDF)")
user_query = st.text_input("Enter a test question:")

if st.button("Run Experiment"):
    files = {"file": uploaded_file.getvalue()}
    # Call your FastAPI backend
    response = requests.post(f"http://localhost:8000/optimize?question={user_query}", files=files)
    data = response.json()


    evaluation_dict = data['evaluation']
    if isinstance(evaluation_dict, str):
        evaluation_dict = json.loads(evaluation_dict)   
        
    st.subheader(f"🏆 Winner: {str(evaluation_dict['winner'])}") 
    st.write(str(evaluation_dict['analysis']))
    
    
    # Show side-by-side comparison
    cols = st.columns(4)
    for i, res in enumerate(data['results']):
        with cols[i]:
            st.info(f"Strategy: {res['name']}")
            st.text_area("Retrieved Context", res['context'], height=200)