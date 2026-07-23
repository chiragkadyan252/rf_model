import os
import joblib
import pandas as pd
import gradio as gr

# Load model
model = joblib.load("car_safety_model.pkl")


def predict_car_safety(buying, maint, doors, persons, lug_boot, safety):

    try:
        input_df = pd.DataFrame([{
            "buying price": buying,
            "maintenance cost": maint,
            "number of doors": doors,
            "number of persons": persons,
            "lug_boot": lug_boot,
            "safety": safety
        }])

        prediction = model.predict(input_df)

        if isinstance(prediction[0], str):

            labels = {
                "unacc": "❌ Unacceptable",
                "acc": "✅ Acceptable",
                "good": "⭐ Good",
                "vgood": "🏆 Very Good"
            }

            return labels.get(prediction[0], prediction[0])

        else:

            labels = {
                0: "❌ Unacceptable",
                1: "✅ Acceptable",
                2: "⭐ Good",
                3: "🏆 Very Good"
            }

            return labels.get(int(prediction[0]), prediction[0])

    except Exception as e:
        return f"Error:\n{e}"


demo = gr.Interface(

    fn=predict_car_safety,

    inputs=[

        gr.Dropdown(
            choices=[("Low",0),("Medium",1),("High",2),("Very High",3)],
            label="Buying Price"
        ),

        gr.Dropdown(
            choices=[("Low",0),("Medium",1),("High",2),("Very High",3)],
            label="Maintenance Cost"
        ),

        gr.Dropdown(
            choices=[
                ("2",0),
                ("3",1),
                ("4",2),
                ("5 or More",3)
            ],
            label="Number of Doors"
        ),

        gr.Dropdown(
            choices=[
                ("2",0),
                ("4",1),
                ("More",2)
            ],
            label="Number of Persons"
        ),

        gr.Dropdown(
            choices=[
                ("Small",0),
                ("Medium",1),
                ("Big",2)
            ],
            label="Luggage Boot"
        ),

        gr.Dropdown(
            choices=[
                ("Low",0),
                ("Medium",1),
                ("High",2)
            ],
            label="Safety"
        )

    ],

    outputs=gr.Textbox(label="Assessment Result"),

    title="🚗 Car Safety Evaluation System",

    description="""
### Developed By
**Chirag Kadyan**

Predict the acceptability of a car using an XGBoost Machine Learning model.
"""
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
