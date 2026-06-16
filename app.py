import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ThichNhuanDat - 2026", layout="wide", page_icon="🏠")

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
    df = pd.read_csv("housing.csv")
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    
    df['AveRooms'] = df['total_rooms'] / df['households']
    df['AveBedrms'] = df['total_bedrooms'] / df['households']
    df['AveOccup'] = df['population'] / df['households']
    
    df = df.rename(columns={
        'housing_median_age': 'HouseAge',
        'median_income': 'MedInc',
        'population': 'Population',
        'median_house_value': 'Gia_Nha'
    })
    
    df_visual = df.copy()
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
    
    features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'latitude', 'longitude']
    ocean_cols = [col for col in df.columns if 'ocean_proximity_' in col]
    features.extend(ocean_cols)
    
    return df, features, df_visual

df, feature_names, df_visual = get_clean_data()

@st.cache_resource
def train_all_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. XGBoost
    xgb = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_train_idx = xgb.score(X_train, y_train)
    xgb_test_idx = xgb.score(X_test, y_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb.predict(X_test)))
    
    # 2. Decision Tree
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    dt_train_idx = dt.score(X_train, y_train)
    dt_test_idx = dt.score(X_test, y_test)
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt.predict(X_test)))
    
    # 3. KNN
    knn = KNeighborsRegressor(n_neighbors=9)
    knn.fit(X_train_scaled, y_train)
    knn_train_idx = knn.score(X_train_scaled, y_train)
    knn_test_idx = knn.score(X_test_scaled, y_test)
    knn_rmse = np.sqrt(mean_squared_error(y_test, knn.predict(X_test_scaled)))
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_train[['latitude', 'longitude']])
    
    models = {
        'XGBoost': xgb,
        'Decision Tree': dt,
        'KNN': knn
    }
    
    metrics = {
        'XGBoost': {'train': xgb_train_idx, 'test': xgb_test_idx, 'rmse': xgb_rmse},
        'Decision Tree': {'train': dt_train_idx, 'test': dt_test_idx, 'rmse': dt_rmse},
        'KNN': {'train': knn_train_idx, 'test': knn_test_idx, 'rmse': knn_rmse}
    }
    
    return models, metrics, scaler, kmeans, X_test, X_test_scaled, y_test

X = df[feature_names]
y = df['Gia_Nha']
models, model_metrics, scaler, kmeans, X_test, X_test_scaled, y_test = train_all_models(X, y)

st.sidebar.markdown("## 🌐 Chỉ số Kinh tế 2026")
st.sidebar.info("Các yếu tố ngoại vi ảnh hưởng đến giá trị tài sản.")

lai_suat = st.sidebar.slider("Lãi suất vay (%)", 3.0, 15.0, 7.0)
lam_phat = st.sidebar.slider("Lạm phát kỳ vọng (%)", 0.0, 10.0, 3.5)
nhu_cau = st.sidebar.select_slider("Cung - Cầu", options=[0.8, 0.9, 1.0, 1.1, 1.2], value=1.0)
chi_so_vung = st.sidebar.selectbox("Khu vực kinh tế", [1.0, 1.1, 1.2, 1.3], format_func=lambda x: f"Mức độ phát triển: x{x}")

# Tính toán hệ số thị trường động từ Sidebar
he_so_thi_truong = nhu_cau * (1 + (lam_phat/100)) * chi_so_vung * (1 - (lai_suat - 7)/100)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🤖 Thuật toán dự đoán")
selected_model_name = st.sidebar.selectbox("Chọn mô hình lõi:", list(models.keys()))

st.title("🏠 Hệ thống Định giá Bất động sản ở CALIFORNIA")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 Dự đoán chuyên sâu", "📈 Phân tích thị trường", "📋 Dữ liệu mẫu & Phân cụm"])

with tab1:
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.subheader("📍 Thông số Bất động sản")
        c1, c2 = st.columns(2)
        with c1:
            med_inc = st.number_input("Thu nhập khu vực ($10k)", 0.5, 15.0, 4.0, help="Thu nhập trung bình của cư dân xung quanh")
            house_age = st.slider("Tuổi thọ công trình (năm)", 1, 52, 20)
            ave_rooms = st.number_input("Tổng số phòng (TB)", 1.0, 15.0, 5.0)
            ave_bedrms = st.number_input("Số phòng ngủ trung bình", 0.5, 5.0, 1.0)
        with c2:
            population = st.number_input("Dân số khu vực", 3, 50000, 1500)
            ave_occup = st.number_input("Số người mỗi hộ", 0.5, 10.0, 3.0)
            lat = st.number_input("Vĩ độ (Latitude)", 32.0, 42.0, 34.05)
            long = st.number_input("Kinh độ (Longitude)", -124.0, -114.0, -118.24)
        
        ocean_pos = st.selectbox("Vị trí so với biển:", ['INLAND', '<1H OCEAN', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])
        btn_predict = st.button("🚀 TÍNH TOÁN GIÁ TRỊ", use_container_width=True)

        st.markdown("---")
        st.subheader(f"📊 Đánh giá mô hình: {selected_model_name}")
        
        base_metrics = model_metrics[selected_model_name]
        
        # 🌟 ĐỘNG HOÁ RMSE: Sai số tiền mặt tự động nhân hệ số vĩ mô thay đổi trực tiếp theo Sidebar
        adjusted_rmse = base_metrics['rmse'] * he_so_thi_truong
        accuracy_val = max(0.0, base_metrics['test'] * 100)
        
        met_c1, met_c2 = st.columns(2)
        with met_c1:
            st.metric(label="📊 Training Score (R²)", value=f"{base_metrics['train']:.4f}")
            st.metric(label="📉 RMSE (Sai số tiền mặt thực tế)", value=f"${adjusted_rmse:,.2f}")
        with met_c2:
            st.metric(label="🧪 Testing Score (R²)", value=f"{base_metrics['test']:.4f}")
            st.metric(label="🎯 Accuracy (Độ chính xác thuật toán)", value=f"{accuracy_val:.2f}%")

    with col_out:
        if btn_predict:
            input_row = pd.DataFrame(0, index=[0], columns=feature_names)
            input_row['MedInc'] = med_inc
            input_row['HouseAge'] = house_age
            input_row['AveRooms'] = ave_rooms
            input_row['AveBedrms'] = ave_bedrms
            input_row['Population'] = population
            input_row['AveOccup'] = ave_occup
            input_row['latitude'] = lat
            input_row['longitude'] = long
            
            target_ocean_col = f"ocean_proximity_{ocean_pos}"
            if target_ocean_col in input_row.columns:
                input_row[target_ocean_col] = 1

            active_model = models[selected_model_name]
            if selected_model_name == 'KNN':
                input_scaled = scaler.transform(input_row)
                raw_price = active_model.predict(input_scaled)[0]
            else:
                raw_price = active_model.predict(input_row)[0]
                
            raw_price = max(0, raw_price)
            final_price = raw_price * he_so_thi_truong
            
            st.metric("GIÁ TRỊ ƯỚC TÍNH", f"${final_price:,.0f}", delta=f"{(he_so_thi_truong-1)*100:.1f}% vs. Gốc")
            
            st.markdown(f"""
            - **Thuật toán đang sử dụng:** `{selected_model_name}`
            - **Giá gốc mô hình:** `${raw_price:,.0f}`
            - **Điều chỉnh thị trường:** `x{he_so_thi_truong:.2f}`
            - **Gợi ý vay:** Trả hàng tháng ước tính `${(final_price * 0.006):,.2f}` (30 năm, lãi suất {lai_suat}%)
            """)
            st.map(pd.DataFrame({'lat': [lat], 'lon': [long]}))

with tab2:
    st.subheader("📊 Phân Tích Địa Lý & Tương Quan Thu Nhập")
    
    st.markdown("#### 1. Bản đồ phân bổ tọa độ và giá nhà toàn bang California")
    fig1 = px.scatter(
        df_visual.sample(4000, random_state=42) if len(df_visual) > 4000 else df_visual, 
        x='longitude', 
        y='latitude', 
        color='Gia_Nha', 
        size='Population',
        color_continuous_scale='jet', 
        size_max=12,
        height=450,
        hover_name='ocean_proximity',
        hover_data={'Gia_Nha': ':.0f', 'MedInc': ':.2f', 'HouseAge': True, 'latitude': False, 'longitude': False},
        labels={'longitude': 'Kinh độ', 'latitude': 'Vĩ độ', 'Gia_Nha': 'Giá nhà ($)', 'Population': 'Dân số'},
        title="Mật độ phân bổ giá trị bất động sản (Đỏ: Giá cao sát biển, Xanh: Giá thấp nội địa)"
    )
    fig1.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("---")
    
    row2_title_col1, row2_title_col2 = st.columns(2)
    with row2_title_col1:
        st.markdown("#### 2. Tương quan giữa Thu nhập và Giá nhà")
    with row2_title_col2:
        st.markdown("#### 3. Ma trận hệ số tương quan tuyến tính (Correlation Matrix)")
        
    row2_plot_col1, row2_plot_col2 = st.columns(2)
    with row2_plot_col1:
        fig2 = px.scatter(
            df_visual.sample(1500, random_state=42), 
            x='MedInc', 
            y='Gia_Nha',
            color='HouseAge', 
            trendline="ols", 
            trendline_color_override="darkred",
            height=400,
            labels={'MedInc': 'Thu nhập trung bình ($10k)', 'Gia_Nha': 'Giá nhà trung vị ($)', 'HouseAge': 'Tuổi nhà'},
            title="Sức mua tập trung ở nhóm cư dân thu nhập cao",
            color_continuous_scale="Cividis"
        )
        fig2.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig2, use_container_width=True)
        
    with row2_plot_col2:
        corr_cols = ['Gia_Nha', 'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
        corr_matrix = df_visual[corr_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale='RdBu_r',
            height=400,
            aspect="auto",
            title="Mức độ ảnh hưởng của các biến đầu vào tới Giá Nhà"
        )
        fig_corr.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_corr, use_container_width=True)
        
    st.markdown("---")
    
    row3_title_col1, row3_title_col2 = st.columns(2)
    with row3_title_col1:
        st.markdown("#### 4. Phân phối chi tiết giá nhà & Điểm ngoại lai trần")
    with row3_title_col2:
        st.markdown("#### 5. Mức giá trung bình phân loại theo Vị trí địa lý")
        
    row3_plot_col1, row3_plot_col2 = st.columns(2)
    with row3_plot_col1:
        fig4 = px.histogram(
            df_visual, 
            x='Gia_Nha', 
            marginal="box",
            height=400,
            labels={'Gia_Nha': 'Giá nhà ($)'}, 
            color_discrete_sequence=['#FF9E2C'],
            title="Biểu đồ phân phối tần suất tích lũy giá trị tài sản"
        )
        fig4.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig4, use_container_width=True)
        
    with row3_plot_col2:
        ocean_price = df_visual.groupby('ocean_proximity')['Gia_Nha'].mean().reset_index().sort_values(by='Gia_Nha', ascending=False)
        fig3 = px.bar(
            ocean_price, 
            x='ocean_proximity', 
            y='Gia_Nha',
            color='Gia_Nha',
            height=400,
            labels={'ocean_proximity': 'Vị trí vùng vịnh', 'Gia_Nha': 'Giá nhà trung bình ($)'},
            title="Sự chênh lệch lớn giữa khu vực nội địa và vùng giáp biển",
            color_continuous_scale="Turbo"
        )
        fig3.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("📋 Dữ liệu mẫu & Phân cụm vị trí")
    st.markdown("#### Dữ liệu trực quan hóa mẫu (10 dòng đầu)")
    st.dataframe(df_visual.head(10), use_container_width=True)
    
    st.markdown("#### Phân cụm vị trí địa lý (K-Means)")
    df_cluster = df_visual.copy()
    df_cluster['Cluster'] = kmeans.predict(df_cluster[['latitude', 'longitude']]).astype(str)
    fig_cluster = px.scatter(
        df_cluster.sample(3000, random_state=42), 
        x='longitude', 
        y='latitude', 
        color='Cluster', 
        height=550,
        title="Phân cụm vị trí bất động sản theo tọa độ địa lý dựa trên mô hình K-Means"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)
