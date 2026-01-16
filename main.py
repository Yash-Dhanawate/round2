import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from utils import calculate_descriptors_safe

st.title("Vascular endothelial growth factor receptor 2 (VEGFR2) Bioactivity Predictor")

st.header("🔬 VEGFR2 (CHEMBL279)")
st.write("""
        • Target: Vascular endothelial growth factor receptor 2 (VEGFR2) \n
        • It is responsible for forming new blood vessels (**angiogenesis**) 
            and maintain a healthy vascular development.\n
        • Upon activation, it triggers signaling pathways 
            that promotes cell growth of blood vessels.\n
        • As it can generate new cells/blood vessels, 
            abnormality might be seen, which can lead to **cancer**.\n
        • Blocking this receptors, it *reduces the blood supply to tumors* 
            and prevents harmful vessel growth.\n
        
""")

df = pd.read_csv(r'data\chemical_data_processed.csv')
with st.expander("🗂️ Dataset:"):
    st.dataframe(df)

corr = df.corr(numeric_only=True)

with st.expander("📟 Correlation Matrix"):
    st.image("plots\correlation_matrix.png", caption="Correlation Matrix")

# ----------------------------------------------------------------
st.divider()
# ----------------------------------------------------------------

st.header("📊 Activity Class Distribution")
st.write("""
         The threshold for activity classification was set at `200 nM`.
         As the  50% of the dataset was below `200 nm` value.
         If a compound has an activity value less than 200 nM, 
         it is classified as `'Active'`, otherwise `'Inactive'`.
         """)

values = df['activity_nM'].dropna()
sns.kdeplot(values, fill=True, log_scale=True)
plt.axvline(x=200, color='red', linestyle='--')
plt.text(210, 0.0025, 'Threshold = 200 nM',
         verticalalignment='bottom', horizontalalignment='left')
plt.title("Activity Value Distribution (nM)")
plt.xlabel("Activity (nM)")
plt.ylabel("Density")
st.pyplot(plt)

# ----------------------------------------------------------------

features = ['Active', 'Inactive']
distribution = df['Activity'].value_counts().to_list()

fig = px.bar(
    df,
    x=features,
    y=distribution,
    color=features, 
    text=distribution,
    title="Activity Class Distribution (Threshold = 200 nM)",
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------
st.divider()
# ----------------------------------------------------------------

st.header("🧠 How Does The Model Work?")

st.markdown("""
                **Input:** Molecular structure (`SMILES`)  
                **Features:** Molecular descriptors & fingerprints  
                **Model:** `Random Forest Classifier`  
                **Output:** `Active` / `Inactive` prediction  

                The model was trained on known bioactivity data from ChEMBL.
                """)
with st.expander("📌 Model Details"):
    st.write("""
                • **Descriptors**: `MW`, `LogP`, `TPSA`, `HBD`, `HBA`, `Rotatable Bonds`, etc.  
                • **Fingerprints**: Morgan (ECFP)  
                • **Threshold**: Activity < `200 nM` = Active
                """)
    st.image(r"plots\feature_importance_Random Forest.png", caption="Feature Importance")
    st.image(r"plots\shap_waterfall_active_12.png", caption="Contribution for Active Prediction")
    st.image(r"plots\shap_waterfall_inactive_12.png", caption="Contribution for Inactive Prediction")

# ----------------------------------------------------------------
st.divider()
# ----------------------------------------------------------------

st.header("🧪 Activity Prediction (Threshold = 200 nM)")
st.write("""
This application predicts the biological activity of chemical compounds
using a pre-trained machine learning model.
It helps chemists quickly evaluate molecules before lab experiments.
""")


smiles = st.text_input("Enter SMILES")
model = pkl.load(open(r'models\best_model(rf_with_fp).pkl', 'rb'))

if st.button("Predict"):
    if smiles:
        descriptors = calculate_descriptors_safe(smiles)
        
        if descriptors is None:
            st.warning("Please enter a valid SMILES")
            st.stop()
        
        prediction = model.predict(descriptors)

        if prediction[0] == 1:
            st.success("Prediction: Active ✅")
        else:
            st.error("Prediction: Inactive ❌")
            
        st.metric("Confidence", f'{model.predict_proba(descriptors).max()*100:.2f}%')
        
        desc_feature_names = ['tpsa', 'mw', 'hbd', 'hba', 'rot', 'logp', 'ring_count']
        st.subheader("🧬 Calculated Descriptors")
        st.dataframe(descriptors[desc_feature_names])
        
        # ------------------------------------------------------------
        
        importances = model.feature_importances_[:len(desc_feature_names)]

        fi_df = pd.DataFrame({
            'Feature': desc_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        st.subheader("🔎 Descriptor Feature Importances")
        px_fig = px.bar(
                        fi_df,
                        x='Importance',
                        y='Feature',
                        title="Descriptor Feature Importances",
                        text='Importance',
                        orientation='h',
                        color='Importance'
        )
        st.plotly_chart(px_fig, use_container_width=True)
    else:
        st.warning("Please enter a valid SMILES")





    



