import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --- Load mô hình và encoder ---
try:
    rf_model, encoder = joblib.load("rfmodel_enc.rpk")
except Exception as e:
    st.error(f"Không thể load mô hình: {e}")
    rf_model, encoder = None, None

# --- Giao diện chính ---
def main():
    st.title("🧠 Internet Addiction Prediction")
    st.write("Nhập thông tin bên dưới để dự đoán mức độ nghiện Internet:")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Giới tính", ["Male", "Female"])
            academic = st.selectbox("Trình độ học vấn", ["Undergraduate", "Graduated", "Highschool"])
            relationship = st.selectbox("Tình trạng mối quan hệ", ["Single", "In a relationship", "Complicated"])
            platform = st.selectbox("Nền tảng sử dụng nhiều nhất", ["Youtube", "Facebook", "TikTok", "Instagram", "Other"])
        with col2:
            sleep_hours = st.number_input("Số giờ ngủ mỗi đêm", min_value=0.0, max_value=12.0, value=7.0)
            mental_health = st.slider("Điểm sức khỏe tâm lý (1-10)", 1, 10, 5)
            usage_hours = st.number_input("Số giờ sử dụng Internet mỗi ngày", min_value=0.0, max_value=24.0, value=5.0)

        submit = st.form_submit_button("🔍 Dự đoán")

    if submit:
        if rf_model is None or encoder is None:
            st.error("Không thể dự đoán vì mô hình hoặc encoder chưa được load đúng cách.")
        else:
        try:
    # --- Dữ liệu gốc (chuỗi) ---
    data = pd.DataFrame([{
        "Gender": gender,
        "Academic_Level": academic,
        "Sleep_Hours_Per_Night": sleep_hours,
        "Relationship_Status": relationship,
        "Mental_Health_Score": mental_health,
        "Most_Used_Platform": platform,
        "Avg_Daily_Usage_Hours": usage_hours
    }])

    st.write("🔎 Dữ liệu gốc (input):")
    st.write(data)

    # --- Kiểm tra encoder tồn tại và kiểu ---
    st.write("🔎 Kiểu encoder:", type(encoder))

    # --- Thử transform và hiển thị kết quả ---
    try:
        X_enc = encoder.transform(data)
    except Exception as e_enc:
        st.error(f"Lỗi khi encoder.transform: {e_enc}")
        st.stop()

    # Nếu là sparse matrix -> convert để in
    try:
        import scipy
        if scipy.sparse.issparse(X_enc):
            X_enc_arr = X_enc.toarray()
        else:
            X_enc_arr = X_enc
    except Exception:
        # nếu scipy không có, cố gắng cast
        X_enc_arr = getattr(X_enc, "toarray", lambda: X_enc)()

    st.write("🔎 Dữ liệu sau encode (array):")
    st.write(X_enc_arr)

    # --- Nếu encoder hỗ trợ get_feature_names_out thì in tên cột encoded ---
    try:
        if hasattr(encoder, "get_feature_names_out"):
            # nếu cần, cung cấp input_features=data.columns
            names = encoder.get_feature_names_out(input_features=data.columns)
            st.write("🔎 Tên features sau encode:")
            st.write(list(names))
        else:
            st.write("🔎 Encoder không có get_feature_names_out()")
    except Exception as e_names:
        st.write("🔎 Lỗi khi lấy tên feature:", e_names)

    # --- In shape để kiểm tra ---
    st.write("🔎 Shapes: data", data.shape, "X_enc", getattr(X_enc_arr, "shape", None))

    # --- Dự đoán ---
    try:
        prediction = rf_model.predict(X_enc)[0]
        st.write("🔎 Dự đoán từ rf_model.predict(X_enc):", prediction)
    except Exception as e_pred:
        st.error(f"Lỗi khi predict: {e_pred}")
        st.stop()

    # --- Làm thêm thử: thay đổi 1 giá trị category để so sánh ---
    data2 = data.copy()
    # đổi giá trị thứ nhất category (nếu có nhiều option, gán 1 option khác)
    # ví dụ đổi Gender sang giá trị khác (nếu hiện là Male -> Female)
    if data2.loc[0, "Gender"] == "Male":
        data2.loc[0, "Gender"] = "Female"
    else:
        data2.loc[0, "Gender"] = "Male"
    st.write("🔁 Dữ liệu gốc chỉnh sửa (data2):")
    st.write(data2)

    try:
        X2_enc = encoder.transform(data2)
        X2_arr = X2_enc.toarray() if scipy.sparse.issparse(X2_enc) else X2_enc
        st.write("🔁 Dữ liệu sau encode (data2):")
        st.write(X2_arr)
        pred2 = rf_model.predict(X2_enc)[0]
        st.write("🔁 Dự đoán data2:", pred2)
    except Exception as e_test:
        st.write("🔁 Lỗi kiểm tra thay đổi:", e_test)

    # --- Hiển thị kết quả chính (tùy bạn dùng prediction hoặc pred2) ---
    final_pred = prediction
    if final_pred < 4:
        level = "Thấp"
    elif final_pred < 7:
        level = "Trung bình"
    else:
        level = "Cao"

    st.success(f"**Điểm dự đoán:** {round(final_pred, 2)}")
    st.info(f"**Mức độ nghiện Internet:** {level}")

except Exception as e:
    st.error(f"Lỗi khi dự đoán: {e}")
