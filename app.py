import os
import joblib
import pandas as pd
import gradio as gr

# ==========================
# Load Trained Model
# ==========================
model = joblib.load("car_safety_model.pkl")

# ==========================
# Mappings (Use the same mappings as during training)
# ==========================
buying_map = {
    "low": 0,
    "med": 1,
    "high": 2,
    "vhigh": 3
}

maint_map = {
    "low": 0,
    "med": 1,
    "high": 2,
    "vhigh": 3
}

doors_map = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5more": 3
}

persons_map = {
    "2": 0,
    "4": 1,
    "more": 2
}

lug_boot_map = {
    "small": 0,
    "med": 1,
    "big": 2
}

safety_map = {
    "low": 0,
    "med": 1,
    "high": 2
}


# ==========================
# Prediction Function
# ==========================
def predict_car_safety(
    buying_price,
    maintenance_cost,
    number_of_doors,
    number_of_persons,
    lug_boot,
    safety,
):
    try:

        df = pd.DataFrame([{
            "buying price": buying_map[buying_price],
            "maintenance cost": maint_map[maintenance_cost],
            "number of doors": doors_map[number_of_doors],
            "number of persons": persons_map[number_of_persons],
            "lug_boot": lug_boot_map[lug_boot],
            "safety": safety_map[safety]
        }])

        prediction = model.predict(df)[0]

        # Optional: convert numeric prediction to label
        label_map = {
            0: "Unacceptable",
            1: "Acceptable",
            2: "Good",
            3: "Very Good"
        }

        return label_map.get(prediction, prediction)

    except Exception as e:
        return f"Error:\n{str(e)}"


# ==========================
# Gradio Interface
# ==========================
demo = gr.Interface(
    fn=predict_car_safety,

    inputs=[
        gr.Dropdown(["vhigh", "high", "med", "low"], label="Buying Price"),
        gr.Dropdown(["vhigh", "high", "med", "low"], label="Maintenance Cost"),
        gr.Dropdown(["2", "3", "4", "5more"], label="Number of Doors"),
        gr.Dropdown(["2", "4", "more"], label="Number of Persons"),
        gr.Dropdown(["small", "med", "big"], label="Luggage Boot"),
        gr.Dropdown(["low", "med", "high"], label="Safety")
    ],

    outputs=gr.Textbox(label="Prediction"),

    title="🚗 Car Safety Prediction System",

    description="""
### Developed By
**Chirag Kadyan**

Enter the car specifications below to predict the safety class using a trained Machine Learning model.
"""
)


# ==========================
# Launch App
# ==========================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
