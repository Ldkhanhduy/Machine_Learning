import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# doc file
doc = pd.read_csv("D:/data/Data_Science_Salary.csv", index_col=None)
# print(doc)



def muc_Luong(file_path, vitri):
    df = doc
    # Chuyển cột 'job_title' về dạng chuỗi để so sánh với giá trị năm cần tìm
    # df['job_title'] = df['job_title'].astype(str)

    # Lọc dữ liệu theo vitri cần tính
    df_vitri = df[df['job_title'] == str(vitri)]
    # Tính tổng số lương trong vị trí cần tính
    tong_so_luong = df_vitri['salary_in_usd'].sum()
    tong_so_luong_lam_tron = round(tong_so_luong, 0)
    # In ra dữ liệu theo năm
    print("year:",df_vitri)
  # In ra tổng số bệnh nhân
    print(f'Tổng số lương trong {vitri}: {tong_so_luong_lam_tron}')
    return tong_so_luong_lam_tron

# muc_Luong(doc,vitri="Data Scientist")

def dem_vitri(file_path, column_name):
    # Đọc file Excel vào DataFrame
    df = doc

    # Đếm số lượng ô chứa giá trị "data_ds" trong cột cụ thể
    dem_o = (df[column_name] == 'Data Scientist').sum()

    # In kết quả
    print(f"Số lượng ô 'Data Scientist' trong cột '{column_name}': {dem_o}")


# dem_vitri(file_path=doc,column_name="job_title")

def dem_congviec(file_path, column_name):
    # Đọc file Excel vào DataFrame
    df = doc

    # Tính số lượng và đếm lặp lại của mỗi giá trị trong cột cụ thể
    value_counts = df[column_name].value_counts()

    # In kết quả
    print(f"{column_name}\tSố lần lặp lại")
    print("-------------------------")
    for value, count in value_counts.items():
        print(f"{value}\t\t{count}")


# dem_congviec(file_path=doc,column_name="job_title")


def tinh_luong_trung_binh(df, position, year):
    # Lọc dữ liệu theo vị trí và năm
    vitri_nam = df[(df['job_title'] == position) & (df['work_year'] == year)]

    # Kiểm tra xem có dữ liệu nào hay không
    if vitri_nam.empty:
        print(f"Không có dữ liệu cho vị trí '{position}' trong năm {year}.")
        return None

    # Tính mức lương trung bình
    luong_trung_binh = vitri_nam['salary_in_usd'].mean()

    # In kết quả
    print(f"Mức lương trung bình cho vị trí '{position}' trong năm {year}: {luong_trung_binh:.2f}")
    
    return luong_trung_binh
# nam 2023
Data_Engineer_2023 = tinh_luong_trung_binh(df=doc,position="Data Engineer",year=2023)
Data_Scientist_2023 = tinh_luong_trung_binh(df=doc,position="Data Scientist",year=2023)
Data_Analys_2023 = tinh_luong_trung_binh(df=doc,position="Data Analyst",year=2023)
Data_Architect_2023 = tinh_luong_trung_binh(df=doc,position="Data Architect",year=2023)
Machine_Learning_Engineer_2023 = tinh_luong_trung_binh(df=doc,position="Machine Learning Engineer",year=2023)
# nam 2022 
Data_Engineer_2022 = tinh_luong_trung_binh(df=doc,position="Data Engineer",year=2022)
Data_Scientist_2022 = tinh_luong_trung_binh(df=doc,position="Data Scientist",year=2022)
Data_Analys_2022 = tinh_luong_trung_binh(df=doc,position="Data Analyst",year=2022)
Data_Architect_2022 =tinh_luong_trung_binh(df=doc,position="Data Architect",year=2022)
Machine_Learning_Engineer_2022 = tinh_luong_trung_binh(df=doc,position="Machine Learning Engineer",year=2022)
# nam 2021
Data_Engineer_2021 = tinh_luong_trung_binh(df=doc,position="Data Engineer",year=2021)
Data_Scientist_2021 = tinh_luong_trung_binh(df=doc,position="Data Scientist",year=2021)
Data_Analys_2021 = tinh_luong_trung_binh(df=doc,position="Data Analyst",year=2021)
Data_Architect_2021 = tinh_luong_trung_binh(df=doc,position="Data Architect",year=2021)
Machine_Learning_Engineer_2021 = tinh_luong_trung_binh(df=doc,position="Machine Learning Engineer",year=2021)
# nam 2020
Data_Engineer_2020 = tinh_luong_trung_binh(df=doc,position="Data Engineer",year=2020)
Data_Scientist_2020 = tinh_luong_trung_binh(df=doc,position="Data Scientist",year=2020)
Data_Analys_2020 = tinh_luong_trung_binh(df=doc,position="Data Analyst",year=2020)
Data_Architect_2020 = tinh_luong_trung_binh(df=doc,position="Data Architect",year=2020)
Machine_Learning_Engineer_2020 = tinh_luong_trung_binh(df=doc,position="Machine Learning Engineer",year=2020)

# ve bieu do line chart
plt.rcParams.update({'font.size': 15})
# Data_Engineer
nam = np.array([2020,2021,2022,2023])
Data_Engineer= Data_Engineer_2020, Data_Engineer_2021,Data_Engineer_2022,Data_Engineer_2023
plt.plot(nam,Data_Engineer, linestyle='-',marker='o', label='Data Engineer')
# Data_Scientist
nam = np.array([2020,2021,2022,2023])
Data_Scientist = Data_Scientist_2020, Data_Scientist_2021,Data_Scientist_2022,Data_Scientist_2023
plt.plot(nam,Data_Scientist, linestyle='-',marker='o', label='Data Scientist')
# Data_Analys
nam = np.array([2020,2021,2022,2023])
Data_Analys = Data_Analys_2020,Data_Analys_2021,Data_Analys_2022,Data_Analys_2023
plt.plot(nam,Data_Analys, linestyle='-',marker='o', label='Data Analys')
# Machine_Learning_Enginee
nam = np.array([2020,2021,2022,2023])
Machine_Learning_Engineer = Machine_Learning_Engineer_2020, Machine_Learning_Engineer_2021, Machine_Learning_Engineer_2022, Machine_Learning_Engineer_2023
plt.plot(nam, Machine_Learning_Engineer, linestyle='-',marker='o', label='Machine Learning Engineer')


plt.xticks(nam)
plt.xlabel('Năm', ha='right', va='top', position=(1, 0))
plt.ylabel('USD', ha='right', va='top', position=(0, 1))
plt.legend()
plt.title('Biểu đồ đường thể hiện mức lương của các công việc theo năm')
plt.show()