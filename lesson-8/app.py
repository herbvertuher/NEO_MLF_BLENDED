import pandas as pd
import joblib
from shiny.express import input, render, ui

pipeline = joblib.load('./final_pipeline_with_model.pkl')

ui.page_opts(title = "Adoption Probability Prediction")

with ui.sidebar(width=500):
    ui.input_selectize("type", "Type", choices=["Cat", "Dog"]),
    ui.input_slider("age", "Age", min=1, max=20, value=5),
    ui.input_selectize("breed1", "Breed1", choices=["Tabby", "Bulldog"]),
    ui.input_selectize("gender", "Gender", choices=["Male", "Female"]),
    ui.input_selectize("color1", "Color1", choices=["Black", "White", "Brown"]),
    ui.input_selectize("color2", "Color2", choices=["None", "White", "Black"]),
    ui.input_selectize("maturity_size", "MaturitySize", choices=["Small", "Medium", "Large"]),
    ui.input_selectize("fur_length", "FurLength", choices=["Short", "Medium", "Long"]),
    ui.input_selectize("vaccinated", "Vaccinated", choices=["No", "Yes"]),
    ui.input_selectize("sterilized", "Sterilized", choices=["No", "Yes"]),
    ui.input_selectize("health", "Health", choices=["Healthy", "Minor Injury", "Serious Injury"]),
    ui.input_slider("fee", "Fee", min=0, max=300, value=50),
    ui.input_text("description", "Description", placeholder="Enter description"),
    ui.input_slider("photo_amt", "PhotoAmt", min=0, max=10, value=1),


@render.ui
def result():

    pet = pd.DataFrame({
        "Type": [input.type()],
        "Age": [input.age()],
        "Breed1": [input.breed1()],
        "Gender": [input.gender()],
        "Color1": [input.color1()],
        "Color2": [input.color2()],
        "MaturitySize": [input.maturity_size()],
        "FurLength": [input.fur_length()],
        "Vaccinated": [input.vaccinated()],
        "Sterilized": [input.sterilized()],
        "Health": [input.health()],
        "Fee": [input.fee()],
        "Description": [input.description()],
        "PhotoAmt": [input.photo_amt()]
    })

    prediction_proba = pipeline.predict_proba(pet)
    adoption_probability = round(prediction_proba[0, 1], 2)

    return f"The probability of adoption is: {adoption_probability:.2%}"