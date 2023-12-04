# Bản đồ vị trí nhân viên trên toàn cầu

# Khai báo các thư viện
import pandas as pd
import country_converter as coco
import plotly.express as px
import warnings

# Tắt thông báo cảnh báo để làm sạch output
warnings.filterwarnings('ignore')


# Tiến hành vẽ bản đồ
def location(df):
    # Chuyển đổi tên quốc gia thành mã ISO3
    country = coco.convert(names=df['employee_residence'], to="ISO3")
    df['employee_residence'] = country

    # Đếm số lượng nhân viên ở mỗi quốc gia
    residence = df['employee_residence'].value_counts()

    # Vẽ bản đồ choropleth
    fig = px.choropleth(locations=(residence.index),
                        color=(residence.values) * 1000,
                        color_continuous_scale=px.colors.sequential.YlGn,
                        title='Bản đồ vị trí nơi ở của nhân viên trên toàn cầu')

    # Cập nhật layout của biểu đồ
    fig.update_layout(
        title_font=dict(size=35, color="#000000")
    )

    # Hiển thị biểu đồ
    fig.show()


# Đọc dữ liệu từ file CSV
df = pd.read_csv("C:/ML/ML1/Data_Science_Salary/DATA/Data_Science_Salary.csv")

# Gọi hàm vẽ bản đồ
location(df)

# Bản đồ phân bổ trung bình lương của ngành KHDL trên toàn cầu
# Hàm chuyển đổi tên quốc gia thành mã ISO3
def convert_to_iso3(df, column_name):
    iso3_codes = coco.convert(names=df[column_name], to="ISO3")
    df[column_name] = iso3_codes


# Hàm nhóm và tính lương trung bình theo vị trí công ty
def calculate_avg_salary_by_location(df):
    grouped_data = df.groupby(['salary_in_usd', 'company_location']).size().reset_index()
    avg_salary_by_location = grouped_data.groupby('company_location').mean().reset_index()
    return avg_salary_by_location


# Hàm vẽ bản đồ choropleth
def draw_choropleth_map(locations, values, title):
    fig = px.choropleth(locations=locations,
                        color=values,
                        title=title,
                        color_continuous_scale=px.colors.sequential.Plasma)

    fig.update_layout(
        title_font=dict(size=35, color="#000000")
    )

    fig.show()


# Đọc dữ liệu từ file CSV
df = pd.read_csv("C:/ML/ML1/Data_Science_Salary/DATA/Data_Science_Salary.csv")

# Chuyển đổi tên quốc gia thành ISO3 code
convert_to_iso3(df, 'company_location')

# Nhóm và tính lương trung bình theo vị trí công ty
average_salary_data = calculate_avg_salary_by_location(df)

# Vẽ bản đồ choropleth
draw_choropleth_map(locations=average_salary_data['company_location'],
                    values=average_salary_data['salary_in_usd'],
                    title='Mức lương trung bình của nhân viên theo vị trí công ty (USD)')