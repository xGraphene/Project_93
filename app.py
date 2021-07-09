import streamlit as st

import model_creation as models

@st.cache
def key_from_val(dict: dict, val: any) -> any:
    keys = []

    for k, v in dict.items():
        if v == val:
            keys.append(k)

    return keys[0] if len(keys) == 1 else keys

@st.cache
def predict(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
    params = [island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]
    model_obj = models.log_reg if model == 'Logistic Regression' else (models.svc_model if model == 'Support Vector Machine' else models.rf_clf)
    predicted_label = model_obj.predict([params])

    return key_from_val(models.SPECIES_MAP, predicted_label)

# create web app
st.sidebar.title('Predict Penguin Species')

model_selection = st.sidebar.selectbox(
    'ML Model', 
    ('Random Forest Classifier', 'Logistic Regression', 'Support Vector Machine')
)

island_selection = st.sidebar.selectbox(
    'Island',
    tuple(models.ISLAND_MAP.keys())
)

sex_selection = st.sidebar.selectbox(
    'Sex',
    tuple(models.SEX_MAP.keys())
)

bill_length_input = st.sidebar.number_input('Bill Length (mm)', value = 0, min_value = 0)
bill_depth_input = st.sidebar.number_input('Bill Depth (mm)', value = 0, min_value = 0)
flipper_length_input = st.sidebar.number_input('Flipper Length (mm)', value = 0, min_value = 0)
body_mass_input = st.sidebar.number_input('Body Magg (g)', value = 0, min_value = 0)

predict_button = st.sidebar.button('Predict')

if predict_button:
    predicted = predict(
        model = model_selection, 
        island = models.ISLAND_MAP[island_selection],
        bill_length_mm = bill_length_input,
        bill_depth_mm = bill_depth_input,
        flipper_length_mm = flipper_length_input,
        body_mass_g = body_mass_input,
        sex = models.SEX_MAP[sex_selection]
    )
    st.write(f'Predicted species: `{predicted}`')
    st.image(f'Images/{predicted}.jpg')
