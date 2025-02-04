import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Vector Database Class
class VectorDatabase:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        self.trials = []
        
    def add_trial(self, trial_data):
        embeddings = self.model.encode(trial_data["criteria"])
        self.index.add(np.array([embeddings]))
        self.trials.append(trial_data)
    
    def search(self, patient_data, threshold=0.85):
        query_embed = self.model.encode(str(patient_data))
        distances, indices = self.index.search(np.array([query_embed]), 5)
        return [
            {**self.trials[i], "score": (1 - distances[0][j]) * 100}
            for j, i in enumerate(indices[0])
            if (1 - distances[0][j]) > threshold
        ]
    
    def count(self):
        return len(self.trials)
    
    def avg_token_count(self):
        return sum(len(trial["criteria"].split()) for trial in self.trials) // len(self.trials)
    
    def list_trials(self):
        return [trial["id"] for trial in self.trials]
    
    def get_trial(self, trial_id):
        return next((trial for trial in self.trials if trial["id"] == trial_id), None)

# LLM Criteria Analyzer Class
class CriteriaAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        self.system_prompt = """
        Analyze oncology clinical trial criteria against patient data.
        Consider diagnosis history, medications, biomarkers, and exclusion factors.
        Return confidence percentage only.
        """
    
    def validate_match(self, patient_data, trial_criteria):
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""
                Patient: {patient_data}
                Criteria: {trial_criteria}
                """}
            ]
        )
        return float(response.choices[0].message.content.strip('%'))
    
    def parse_criteria(self, criteria):
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Parse and structure clinical trial criteria."},
                {"role": "user", "content": f"Analyze and structure this criteria: {criteria}"}
            ]
        )
        return response.choices[0].message.content

# Initialize components
vector_db = VectorDatabase()
llm_analyzer = CriteriaAnalyzer()

# Streamlit App
st.title("Oncology Clinical Trial Matcher 🩺🔍")
st.markdown("""
**Match patient records to clinical trials using AI-powered analysis of inclusion/exclusion criteria**
""")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    trial_data_source = st.selectbox(
        "Trial Data Source",
        ["Vector Database", "CSV Upload"]
    )
    similarity_threshold = st.slider(
        "Match Threshold (%)", 70, 100, 85
    )

# Main interface
tab1, tab2 = st.tabs(["Patient Upload", "Trial Management"])

with tab1:
    # Patient data upload
    uploaded_file = st.file_uploader(
        "Upload Patient CSV Records",
        type=["csv"],
        help="Requires columns: patient_id, diagnosis, medications, age, biomarkers"
    )
    
    if uploaded_file:
        patient_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(patient_df)} patient records")
        
        if st.button("Analyze Eligibility"):
            with st.status("Processing..."):
                results = []
                for _, patient in patient_df.iterrows():
                    # Vector similarity search
                    trial_matches = vector_db.search(
                        patient.to_dict(),
                        threshold=similarity_threshold/100
                    )
                    
                    # LLM validation
                    for trial in trial_matches:
                        eligibility = llm_analyzer.validate_match(
                            patient_data=patient,
                            trial_criteria=trial["criteria"]
                        )
                        results.append({
                            **patient,
                            "trial_id": trial["id"],
                            "match_score": trial["score"],
                            "eligibility": eligibility
                        })
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(
                    results_df,
                    column_config={
                        "eligibility": st.column_config.ProgressColumn(
                            "Confidence",
                            format="%.0f%%",
                            min_value=0,
                            max_value=100
                        )
                    }
                )

with tab2:
    # Trial criteria management
    st.header("Clinical Trial Database")
    
    # Vector DB statistics
    col1, col2 = st.columns(2)
    col1.metric("Total Trials", vector_db.count())
    col2.metric("Avg Criteria Complexity", f"{vector_db.avg_token_count()} tokens")
    
    # Trial inspection
    selected_trial = st.selectbox(
        "View Trial Details",
        vector_db.list_trials()
    )
    if selected_trial:
        trial_data = vector_db.get_trial(selected_trial)
        with st.expander("Criteria Analysis"):
            st.json(llm_analyzer.parse_criteria(trial_data["criteria"]))

# Sample data initialization (for demonstration purposes)
if 'initialized' not in st.session_state:
    sample_trials = [
        {"id": "NCT001", "criteria": "Adult patients with stage III or IV non-small cell lung cancer. No prior chemotherapy."},
        {"id": "NCT002", "criteria": "Breast cancer patients with HER2-positive tumors. Prior treatment with trastuzumab allowed."},
        {"id": "NCT003", "criteria": "Colorectal cancer patients with KRAS wild-type tumors. No active brain metastases."}
    ]
    for trial in sample_trials:
        vector_db.add_trial(trial)
    st.session_state.initialized = True

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.info("This is a demo application. In a production environment, ensure proper data security and HIPAA compliance.")
