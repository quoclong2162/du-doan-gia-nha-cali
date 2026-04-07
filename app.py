import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import joblib
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(page_title="ThichNhuanDat", layout="wide", page_icon="🏠")


def add_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), 
                              url("https://static0.thetravelimages.com/wordpress/wp-content/uploads/2024/09/resized-image-16.jpg?w=1600&h=900&fit=crop");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
        }

        h1 {
            color: #FFFFFF !important;
            text-shadow: 3px 3px 6px #2D1B4E;
            background-color: rgba(45, 27, 78, 0.5);
            padding: 15px 25px;
            border-radius: 15px;
            border-left: 5px solid #FF9E2C;
        }

        [data-testid="stSidebar"] {
            background-color: rgba(15, 15, 25, 0.98) !important;
            border-right: 1px solid rgba(255, 158, 44, 0.3);
        }

        [data-testid="stSidebar"] .stMarkdown h2, 
        [data-testid="stSidebar"] label p {
            color: #FFB347 !important;
        }

        [data-testid="stMetric"] {
            background-color: rgba(28, 33, 53, 0.85) !important; 
            border: 1px solid rgba(255, 158, 44, 0.4);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        }

        [data-testid="stMetricLabel"] p {
            color: #E0E0E0 !important;
        }

        [data-testid="stMetricValue"] div {
            color: #FFB347 !important; 
            text-shadow: 0px 0px 10px rgba(255, 179, 71, 0.4);
        }

        .stTabs [data-baseweb="tab-list"] {
            background-color: rgba(28, 33, 53, 0.95);
            border-radius: 12px;
            padding: 8px;
        }

        button[data-baseweb="tab"] p {
            color: #FFB347 !important;
        }
        
        .stDataFrame, .js-plotly-plot {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 15px;
            border: 1px solid #FF9E2C;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

add_custom_style()

@st.cache_data
def get_clean_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Gia_Nha'] = housing.target
    
    df = df[df['Gia_Nha'] < 4.9] 
    return df, housing.feature_names

df, feature_names = get_clean_data()


ten_tieng_viet = {
    'MedInc': 'Thu nhập TB', 'HouseAge': 'Tuổi nhà', 'AveRooms': 'Số phòng TB',
    'AveBedrms': 'Phòng ngủ TB', 'Population': 'Dân số', 'AveOccup': 'Số người/Hộ',
    'Latitude': 'Vĩ độ', 'Longitude': 'Kinh độ'
}

@st.cache_resource
def train_optimized_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

X = df.drop('Gia_Nha', axis=1)
y = df['Gia_Nha']
model, X_test, y_test = train_optimized_model(X, y)


st.sidebar.markdown("## 🌐 Chỉ số Kinh tế 2026")
st.sidebar.info("Các yếu tố ngoại vi ảnh hưởng đến giá trị tài sản.")

lai_suat = st.sidebar.slider("Lãi suất vay (%)", 3.0, 15.0, 7.0)
lam_phat = st.sidebar.slider("Lạm phát kỳ vọng (%)", 0.0, 10.0, 3.5)
nhu_cau = st.sidebar.select_slider("Cung - Cầu", options=[0.8, 0.9, 1.0, 1.1, 1.2], value=1.0)
chi_so_vung = st.sidebar.selectbox("Khu vực kinh tế", [1.0, 1.1, 1.2, 1.3], format_func=lambda x: f"Mức độ phát triển: x{x}")


he_so_thi_truong = nhu_cau * (1 + (lam_phat/100)) * chi_so_vung * (1 - (lai_suat - 7)/100)


st.title("🏠 Hệ thống Định giá Bất động sản ở CALIFORNIA")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 Dự đoán chuyên sâu", "📈 Phân tích thị trường", "📋 Dữ liệu mẫu"])

with tab1:
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.subheader("📍 Thông số Bất động sản")
        c1, c2 = st.columns(2)
        with c1:
            med_inc = st.number_input("Thu nhập khu vực ($10k)", 0.5, 15.0, 4.0, help="Thu nhập trung bình của cư dân xung quanh")
            house_age = st.slider("Tuổi thọ công trình (năm)", 1, 52, 20)
            ave_rooms = st.number_input("Tổng số phòng (TB)", 1.0, 15.0, 5.0)
        with c2:
            ave_occup = st.number_input("Số người mỗi hộ", 0.5, 10.0, 3.0)
            lat = st.number_input("Vĩ độ (Latitude)", 32.0, 42.0, 34.05)
            long = st.number_input("Kinh độ (Longitude)", -124.0, -114.0, -118.24)
        
        btn_predict = st.button("🚀 TÍNH TOÁN GIÁ TRỊ", use_container_width=True)

    with col_out:
        if btn_predict:
            
            input_data = pd.DataFrame([[med_inc, house_age, ave_rooms, 1.0, 1500, ave_occup, lat, long]], columns=feature_names)
            
            raw_price = model.predict(input_data)[0]
            final_price = raw_price * he_so_thi_truong * 100000 
            
            st.metric("GIÁ TRỊ ƯỚC TÍNH", f"${final_price:,.0f}", delta=f"{(he_so_thi_truong-1)*100:.1f}% vs. Gốc")
            
            
            st.markdown(f"""
            - **Giá gốc mô hình:** `${raw_price*100000:,.0f}`
            - **Điều chỉnh thị trường:** `x{he_so_thi_truong:.2f}`
            - **Gợi ý vay:** Trả hàng tháng ước tính `${(final_price * 0.006):,.2f}` (30 năm, lãi suất {lai_suat}%)
            """)
            st.map(pd.DataFrame({'lat': [lat], 'lon': [long]}))

with tab2:
    st.subheader("📊 Hiệu suất Mô hình & Đặc trưng")
    m1, m2 = st.columns(2)
    
    
    feat_imp = pd.DataFrame({'Yếu tố': [ten_tieng_viet.get(f, f) for f in feature_names], 'Độ ảnh hưởng': model.feature_importances_})
    feat_imp = feat_imp.sort_values('Độ ảnh hưởng', ascending=True)
    fig_imp = px.bar(feat_imp, x='Độ ảnh hưởng', y='Yếu tố', orientation='h', title="Yếu tố nào quyết định giá?")
    m1.plotly_chart(fig_imp, use_container_width=True)
    
    
    y_pred = model.predict(X_test) * he_so_thi_truong
    fig_res = px.scatter(x=y_test * 100000, y=y_pred * 100000, labels={'x': 'Giá thật', 'y': 'Giá dự đoán'}, title="Độ chính xác thực tế")
    fig_res.add_shape(type="line", x0=0, y0=0, x1=500000, y1=500000, line=dict(color="Red"))
    m2.plotly_chart(fig_res, use_container_width=True)

with tab3:
    st.dataframe(df.head(100), use_container_width=True)
    st.download_button("Tải dữ liệu sạch (.csv)", df.to_csv().encode('utf-8'), "cali_housing_clean.csv")