import streamlit as st
import pandas as pd
import joblib


#Load Model Files
model=joblib.load("profit_prediction_model.pkl")
scaler=joblib.load("scaler.pkl")
feature_names=joblib.load("features.pkl")



st.set_page_config(
    page_title="Sales Profit Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Profit Prediction App")
st.write("Predict Profit Based on Sales Details")
st.markdown(
"""
Predict product profit based on sales features using a Machine Learning model.

✅ Random Forest Model  
✅ Feature Engineering  
✅ Real Business Dataset
"""
)


#User input
sales=st.number_input("Sales",0.0)
quantity=st.number_input("Quantity",1)
discount=st.slider("Discount",0.0,1.0)
shipping_day=st.number_input("Shipping Day",0)
order_month=st.slider("Order Month",1,12)
order_year=st.number_input("Order Year",2015)

#Creqate input data
input_dict={col:0 for col in feature_names}
input_dict["Sales"]=sales
input_dict["Quantity"]=quantity
input_dict["Discount"]=discount
input_dict["Shipping_Days"]=shipping_day
input_dict["Order_Month"]=order_month
input_dict["Order_Year"]=order_year

input_df=pd.DataFrame([input_dict])

#Scale input
input_scaled=scaler.transform(input_df)

#Prediction
if st.button("Predict Profit"):
    prediction=model.predict(input_scaled)
    st.success(f"Estimated profit: ${prediction[0]:.2f}")


st.subheader("📈 Model Feature Importance")
st.image("feature_importance.png")