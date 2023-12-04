import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# doc file
doc = pd.read_csv("D:/data/Data_Science_Salary.csv", index_col=None)
# print(doc)
def dem_vung(file_path, column_name):
    # Đọc file Excel vào DataFrame
    df = doc

    # Tính số lượng và đếm lặp lại của mỗi giá trị trong cột cụ thể
    value_counts = df[column_name].value_counts()

    # In kết quả
    print(f"{column_name}\tSố lần lặp lại")
    print("-------------------------")
    for value, count in value_counts.items():
        print(f"{value}\t\t{count}")

dem_vung(file_path=doc, column_name="employee_residence")

#

def dem_cong_viec(df):
    # Groupby theo cột "Địa điểm" và đếm số lượng công việc A
    result = df[df['job_title'] == 'Data Engineer'].groupby('employee_residence').size()

    # In kết quả
    print("Cong viec tai:",result)

# dem_cong_viec(df=doc)    


def dem_so_luong_cong_viec_theo_vung(df,job, location):
    # Lọc dữ liệu theo điều kiện công việc là 'A' và địa điểm là location
    cong_viec = df[(df['job_title'] == job) & (df['employee_residence'] == location)]

    # Đếm số lượng công việc A cho địa điểm cụ thể
    count = len(cong_viec)

    # In kết quả
    print(f"Số lượng công việc {job} cho vùng {location}: {count}")

# đếm số lượng công việc ở US
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Engineer", location="US")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Scientist", locatiAn="US")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Analys", location="US")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Architect", location="US")
# # đếm số lượng công việc ở GB
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Engineer", location="GB")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Scientist", location="GB")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Analys", location="GB")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Architect", location="GB")

# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Engineer", location="CA")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Scientist", location="CA")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Analys", location="CA")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Architect", location="CA")

# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Engineer", location="ES")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Scientist", location="ES")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Analys", location="ES")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Architect", location="ES")

# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Engineer", location="IN")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Scientist", location="IN")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Analys", location="IN")
# dem_so_luong_cong_viec_theo_vung(df=doc,job="Data Architect", location="IN")

# tính tổng mức lương của job theo vùng

def trung_binh_luong_cong_viec_theo_vung(df, job, location):
    # Lọc dữ liệu theo công việc và địa điểm
    job_area = df[(df['job_title'] == job) & (df['company_location'] == location)]

    # Tính tổng mức lương
    trungbinh_luong = job_area['salary_in_usd'].mean()
    # Làm tròn
    luong_mean = round(trungbinh_luong,0)
    return luong_mean
    # In kết quả 
    print(f"Trung bình mức lương cho công việc {job} ở địa điểm {location}: {luong_mean}")
print("Data Engineer")
DE_US = trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Data Engineer", location="US")
DE_GB =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Data Engineer", location="GB")
DE_CA =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Data Engineer", location="CA")
DE_ES =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Data Engineer", location="ES") 
DE_IN =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Data Engineer", location="IN") 
print("Data Scientist")
DS_US =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Data Scientist", location="US")
DS_GB =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Data Scientist", location="GB")
DS_CA =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Data Scientist", location="CA")
DS_ES =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Data Scientist", location="ES") 
DS_IN =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Data Scientist", location="IN") 
print("Machine Learning Engineer")
MLE_US =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Machine Learning Engineer", location="US")
MLE_GB =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Machine Learning Engineer", location="GB")
MLE_CA =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Machine Learning Engineer", location="CA")
MLE_ES =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Machine Learning Engineer", location="ES") 
MLE_IN =trung_binh_luong_cong_viec_theo_vung(df=doc,job= "Machine Learning Engineer", location="IN") 
 

#  vẽ biểu đồ 
plt.rcParams.update({'font.size': 15})
# Dữ liệu
area = ["United States", "UK and Northern Ireland", "Canada", "Spain", "India"]
DE = DE_US, DE_GB, DE_CA, DE_ES, DE_IN
DS = DS_US, DS_GB, DS_CA, DS_ES, DS_IN
MLE = MLE_US, MLE_GB, MLE_CA, MLE_ES, MLE_IN

# Vẽ biểu đồ vùng 
plt.stackplot(area,DE, DS,MLE, labels=['Data Engineer', 'Data Scientist', 'Data Analyst', 'Data Architect', 'Machine Learning Engineer'], colors=['#f7b538', '#297373', '#780116', 'orange', 'purple'], alpha=0.4)
plt.title('Biểu đồ thể hiện mức lương trung bình các công việc theo vùng/quốc gia')
plt.xlabel('Vùng/Quốc gia', ha='right', va='top', position=(1, 0))
plt.ylabel('USD', ha='right', va='top', position=(0, 1))
plt.legend()
plt.grid(True)
# Thêm dữ liệu cụ thể 
for i, (a, de, ds, mle) in enumerate(zip(area, DE, DS, MLE)):
    plt.annotate(f" {de}\n {ds}\n {mle}", (i, (de + ds + mle) / 2), ha='center', va='center', fontsize=8, color='black')
plt.show()
