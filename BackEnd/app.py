from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from datetime import date
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OneHotEncoder
import google.generativeai as genai



# -------------------------------
# Gemini API Setup
# -------------------------------
GEMINI_API_KEY = "AIzaSyACDtZN07y2QhaV3X6x9B5V_yxTZgkLlIE"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Loading Model...")

model = pickle.load(open(r"C:\Users\Computec\Desktop\lotfy\BackEnd\model.pkl", "rb"))

FINAL_COLUMNS = [
'City_0','City_1','City_2','City_3','City_4','City_5','City_6','City_7','City_8',
'State_0','State_1','State_2','State_3','State_4','State_5','State_6','State_7','State_8',
'Country_0','Country_1','Country_2','Country_3','Country_4','Country_5','Country_6',
'Product Name_0','Product Name_1','Product Name_2','Product Name_3','Product Name_4',
'Product Name_5','Product Name_6','Product Name_7','Product Name_8','Product Name_9',
'Product Name_10','Product Name_11','Quantity','year','month','day','day_of_week',
'is_weekend','lb_Order Priority','Segment_Consumer','Segment_Corporate',
'Segment_Home Office','Ship Mode_First Class','Ship Mode_Same Day',
'Ship Mode_Second Class','Ship Mode_Standard Class','Region_Africa','Region_Canada',
'Region_Caribbean','Region_Central','Region_Central Asia','Region_EMEA','Region_East',
'Region_North','Region_North Asia','Region_Oceania','Region_South',
'Region_Southeast Asia','Region_West','Category_Furniture',
'Category_Office Supplies','Category_Technology','Sub-Category_Accessories',
'Sub-Category_Appliances','Sub-Category_Art','Sub-Category_Binders',
'Sub-Category_Bookcases','Sub-Category_Chairs','Sub-Category_Copiers',
'Sub-Category_Envelopes','Sub-Category_Fasteners','Sub-Category_Furnishings',
'Sub-Category_Labels','Sub-Category_Machines','Sub-Category_Paper',
'Sub-Category_Phones','Sub-Category_Storage','Sub-Category_Supplies',
'Sub-Category_Tables'
]

class InputData(BaseModel):
    City: str
    State: str
    Country: str
    ProductName: str
    OrderPriority: str
    Segment: str
    ShipMode: str
    Region: str
    Category: str
    SubCategory: str
    Quantity: int
    OrderDate: date


class ChatMessage(BaseModel):
    message: str


@app.post("/chat")
def chat_with_ai(data: ChatMessage):
    try:
        response = gemini_model.generate_content(data.message)
        return {"reply": response.text}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
def predict(data: InputData):

    # ---------- 1) DATE FEATURES ----------
    year = data.OrderDate.year
    month = data.OrderDate.month
    day = data.OrderDate.day
    day_of_week = data.OrderDate.weekday()
    is_weekend = 1 if day_of_week in [5, 6] else 0

    # ---------- 2) PRIORITY MAPPING ----------
    priority_map = {"Low":0, "Medium":1, "High":2, "Critical":3}
    pr_value = priority_map[data.OrderPriority]

    # ---------- 3) CREATE DF ----------
    df = pd.DataFrame([{
        "City": data.City,
        "State": data.State,
        "Country": data.Country,
        "Product Name": data.ProductName,
        "Segment": data.Segment,
        "Ship Mode": data.ShipMode,
        "Region": data.Region,
        "Category": data.Category,
        "Sub-Category": data.SubCategory,
        "Order Priority": data.OrderPriority,
        "Quantity": data.Quantity,
        "year": year,
        "month": month,
        "day": day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "lb_Order Priority": pr_value
    }])

    # ---------- 4) Binary Encoding ----------
    bin_encoder = BinaryEncoder(cols=['City','State','Country','Product Name'])
    bin_encoded = bin_encoder.fit_transform(df)

    # ---------- 5) OneHot Encoding ----------
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    nominal_cols = ['Segment','Ship Mode','Region','Category','Sub-Category']
    ohe_encoded = pd.DataFrame(
        ohe.fit_transform(df[nominal_cols]),
        columns=ohe.get_feature_names_out(nominal_cols)
    )

    # ---------- 6) Merge ----------
    merged = pd.concat([bin_encoded, ohe_encoded], axis=1)

    drop_cols = ['Ship Mode','Segment','Region','Category','Sub-Category','Order Priority','Market']
    for c in drop_cols:
        if c in merged:
            merged = merged.drop(columns=[c])

    # ---------- 7) Align ----------
    for col in FINAL_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0

    merged = merged[FINAL_COLUMNS]

    # ---------- 8) Predict ----------
    pred = model.predict(merged)[0]

    # -------------------------------
    # ✅ 9) INVERSE LOG HERE
    # -------------------------------
    sales = np.expm1(pred[0])          # undo log1p
    shipping = np.expm1(pred[1])       # undo log1p
    discount = pred[2]                 # discount stays as is

    # ---------- RETURN ----------
    return {
        "sales": float(sales),
        "shipping_cost": float(shipping),
        "discount": float(discount)
    }
