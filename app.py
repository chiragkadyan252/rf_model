import gradio as gr
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("car_safety_model.pkl")


def predict_car_safety(buying_price, maintenance_cost, number_of_doors,
                       number_of_persons, lug_boot, safety):

    input_data = pd.DataFrame([{
        "buying price": buying_price,
        "maintenance cost": maintenance_cost,
        "number of doors": number_of_doors,
        "number of persons": number_of_persons,
        "lug_boot": lug_boot,
        "safety": safety
    }])

    prediction = model.predict(input_data)[0]

    return f"Predicted Class: {prediction}"


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
    **Developed by:** Chirag Kadyan  
    **Roll No.:** 241513

    Enter the car specifications below to predict its safety class using a trained Machine Learning model.
    """
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
