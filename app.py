from dash import Dash, html, dcc, Input, Output
import pandas as pd
import numpy as np
import joblib

# Load the trained pipeline model
model = joblib.load('selling_price2.model')

# Initialize the Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Car Price Prediction", style={"textAlign": "center", "marginBottom": "30px"}),
    html.Div([
        html.Div([
            html.Label("Year", style={"fontWeight": "bold"}),
            dcc.Input(id="year", type="number", placeholder="Year", min=1900, max=2100, style={"width": "100%", "marginBottom": "15px"}),
            
            html.Label("Mileage (km/l)", style={"fontWeight": "bold"}),
            dcc.Input(id="mileage", type="number", placeholder="Mileage", style={"width": "100%", "marginBottom": "15px"}),
            
            html.Label("KM Driven", style={"fontWeight": "bold"}),
            dcc.Input(id="km_driven", type="number", placeholder="KM Driven", style={"width": "100%", "marginBottom": "15px"}),
            
            html.Label("Owner (1=First, 2=Second, 3=Third, 4=Fourth+)", style={"fontWeight": "bold"}),
            dcc.Input(id="owner", type="number", placeholder="Owner", min=1, max=4, style={"width": "100%", "marginBottom": "25px"}),
            
            html.Button("Predict", id="predict-button", n_clicks=0, style={"width": "100%", "marginBottom": "20px"}),
            html.Div(id="result", style={"marginTop": "20px", "fontSize": "20px", "textAlign": "center"})
        ], style={
            "background": "#f9f9f9",
            "padding": "30px",
            "borderRadius": "10px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.1)"
        })
    ], style={
        "maxWidth": "400px",
        "margin": "auto"
    })
])

@app.callback(
    Output("result", "children"),
    [
        Input("predict-button", "n_clicks"),
        Input("year", "value"),
        Input("mileage", "value"),
        Input("km_driven", "value"),
        Input("owner", "value"),
    ]
)
def predict_price(n_clicks, year, mileage, km_driven, owner):
    if n_clicks > 0:
        try:
            if year is None or mileage is None or km_driven is None or owner is None:
                return "Please fill in all fields before predicting."

            input_data = pd.DataFrame({
                "year": [year],
                "mileage": [mileage],
                "km_driven": [km_driven],
                "owner": [owner],
            })

            raw_pred = model.predict(input_data)[0]
            predicted_price = np.exp(raw_pred)

            return f"The predicted car price is ≈ {predicted_price:,.2f} Baht"
        except Exception as e:
            return f"An error occurred: {e}"
    return "Click the predict button to see the result."

if __name__ == "__main__":
    app.run(debug=True)