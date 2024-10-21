import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# buoc 1: Tai du lieu 
data = pd.read_csv('ML.csv')

# Kiem tra cac gia tri NaN
print("Số lượng giá trị NaN trong mỗi cột:")
print(data.isnull().sum())

# Xu ly gia tri NaN
data = data.dropna() 

# Chia du lieu thanh tap train (80%) và tap test (20%)
X = data.drop('College GPA', axis=1)  # Các đặc trưng
y = data['College GPA']  # Điểm số cần dự đoán

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Tiền xử lý dữ liệu
categorical_features = ['Hobby Activities', 'Employment Status']
numerical_features = ['Hours Studied Weekly', 'Class Participation Rate', 'Secondary School GPA', 'Online Research Hours', 'Age']

# Tạo pipeline cho tiền xử lý dữ liệu
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),  # Cột số
        ('cat', OneHotEncoder(), categorical_features)  # Cột phân loại
    ])

# Áp dụng tiền xử lý
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#  Áp dụng mô hình hồi quy tuyến tính
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Dự đoán điểm số
y_pred_lin = lin_reg.predict(X_test)

# Đánh giá mô hình hồi quy tuyến tính
mae_lin = mean_absolute_error(y_test, y_pred_lin)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

# Bước 5: Áp dụng mô hình Decision Tree Regression
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

# Dự đoán điểm số
y_pred_tree = tree_reg.predict(X_test)

# Đánh giá mô hình Decision Tree
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

# Tạo bảng kết quả
results = pd.DataFrame({
    'Thamso': ['MAE', 'MSE', 'R-squared'],
    'Linear Regression': [mae_lin, mse_lin, r2_lin],
    'Decision Tree Regression': [mae_tree, mse_tree, r2_tree]
})

# In ra bảng kết quả
print('\n--- KẾT QUẢ ---')
print(results)
