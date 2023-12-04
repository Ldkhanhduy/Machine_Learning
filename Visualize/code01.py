import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb

DSS = pd.read_csv("D:/data/Data_Science_Salary.csv", index_col=None)
pd.set_option('display.max_columns', None)
def stack_bar_chart():
    salary_by_level = DSS.pivot_table(index='experience_level', columns='employment_type', values='salary_in_usd', aggfunc=np.mean)
    print(salary_by_level.fillna(0))
    plt.bar(salary_by_level.index, salary_by_level['FT'])
    plt.bar(salary_by_level.index, salary_by_level['CT'], bottom=salary_by_level['FT'], color='red')
    plt.bar(salary_by_level.index, salary_by_level['FL'], bottom=salary_by_level['FT']+salary_by_level['CT'], color='green')
    plt.bar(salary_by_level.index, salary_by_level['PT'], bottom=salary_by_level['FT']+salary_by_level['CT']+salary_by_level['FL'], color='black')
    plt.legend(['FT','CT','FL','PT'], fontsize= 20)
    plt.grid(True)
    plt.title("Biểu đồ trung bình lương theo trình độ", fontsize= 20)
    plt.xlabel("Trình độ", fontsize= 20)
    plt.ylabel("USD", fontsize= 20)
    plt.show()
def doughnut_chart():
    size_company = pd.crosstab(index=DSS['company_size'], columns=DSS['experience_level'])
    print(size_company)
    fig, ax = plt.subplots(figsize=(7,10))
    ax.pie([size_company.iloc[0].sum(),size_company.iloc[1].sum(),size_company.iloc[2].sum()], colors=['#FFD700','#FFED97','#FFFACD'], radius=0.7,autopct='%1.0f%%', wedgeprops=dict(width=0.3))
    ax.pie(list(size_company.iloc[0])+list(size_company.iloc[1])+list(size_company.iloc[2]),colors=['#ADD8E6','#87CEFA','#6495ED','#4169E1'], radius=1.0, wedgeprops=dict(width=0.3), labels=['22%','2.9%','29.5','45.6%','5.4%','3.2%','19.6%','71.8%','33.1%','4.1%','33.7%','29.1%'])
    centre_circle = plt.Circle((0, 0), 0.30, fc='white')
    plt.title('BIỂU ĐỒ TRÒN BIỂU THỊ TRÌNH ĐỘ NHÂN VIÊN THEO QUY MÔ CÔNG TY',fontsize=20)
    plt.legend(['L','M','S','EN','EX','MI','SE'], fontsize=20, loc='lower center', ncol=7)
    fig.gca().add_artist(centre_circle)
    plt.show()
def histogram_density_chart():
    fig, ax = plt.subplots()
    ax.hist(DSS['salary_in_usd'], bins=20, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Biểu đồ phân phối tần suất')
    sb.kdeplot(DSS['salary_in_usd'], color='green', label='Biểu đồ mật độ')
    plt.title('Biểu đồ tần suất về thu nhập của các nhà khoa học dữ liệu', fontsize=20)
    plt.xlabel('Thu nhập (USD)', fontsize=20)
    plt.ylabel('Tần suất', fontsize=20)
    plt.legend(fontsize=20)
    plt.show()
# stack_bar_chart()
# doughnut_chart()
def area_chart():
    work_type = pd.crosstab(index=DSS['work_year'], columns=DSS['remote_ratio'])
    for i in range(0,4):
        work_type.iloc[i] = work_type.iloc[i]/work_type.iloc[i].sum()*100
    print(work_type)
    plt.fill_between(work_type.index, np.array(work_type[0]),[0,0,0,0], color= '#FFB6C1', alpha=0.7)
    plt.fill_between(work_type.index, np.array(work_type[0]), np.array(work_type[0])+np.array(work_type[50]), color='#E6E6FA', alpha=0.7)
    plt.fill_between(work_type.index, np.array(work_type[0])+np.array(work_type[50]), np.array(work_type[0])+np.array(work_type[50])+np.array(work_type[100]), color='#FF8C00', alpha=0.7)
    plt.legend(['Offline', '50-50', 'Online'], fontsize=20)
    plt.title('Biểu đồ miền biểu diễn tỉ lệ làm việc trực tuyến, trực tiếp qua các năm', fontsize=20)
    plt.ylabel("%", fontsize=20, rotation=0)
    plt.grid(alpha=0.4)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Year", fontsize=20)
    plt.show()
def scatter_plot():
    data1 = DSS[DSS['job_title'] == 'Machine Learning Engineer']
    data2 = DSS[DSS['job_title'] == 'Analytics Engineer']
    plt.scatter(data1['remote_ratio'], data1['salary_in_usd'], label='Machine Learning Engineer')
    plt.scatter(data2['remote_ratio'], data2['salary_in_usd'], label='Analytics Engineer')
    plt.legend(fontsize=15)
    plt.grid(alpha=0.4)
    plt.xlabel("Tỉ lệ làm việc từ xa (%)", fontsize=20)
    plt.ylabel("Mức thu nhập (USD)", fontsize=20)
    plt.title("Biểu đồ phân tán biểu thị mức thu nhập theo tỉ lệ làm việc trực tuyến", fontsize=20)
    plt.show()
def pie_chart():
    salary_cur1 = DSS[DSS['salary_currency']=='USD']
    salary_cur2 = DSS[DSS['salary_currency']=='EUR']
    fig, ax = plt.subplots(1,2)
    ax[0].pie(salary_cur1['company_size'].value_counts(), autopct='%1.0f%%', colors=['#00FF00','#228B22','#90EE90'])
    ax[1].pie(salary_cur2['company_size'].value_counts(), autopct='%1.0f%%', colors=['#00FF00','#228B22','#90EE90'] )
    ax[0].set_title("USD", fontsize=20,loc="center")
    ax[1].set_title("EUR", fontsize=20,loc="center")
    fig.suptitle("Biểu đồ tròn biểu thị tỉ lệ quy mô công ty sử dụng mệnh giá tiền",fontsize=20)
    fig.legend(['Quy mô nhỏ', 'Quy mô vừa', 'Quy mô lớn'], fontsize=20, loc='lower center', ncol=3)
    plt.show()
def buble_chart():
    salary_cur1 = DSS[DSS['salary_currency']=='USD']
    salary_cur2 = DSS[DSS['salary_currency']=='EUR']
    usd1 = salary_cur1.pivot_table(index='salary_currency', columns='remote_ratio', values='salary_in_usd', aggfunc=sum)
    usd2 = salary_cur2.pivot_table(index='salary_currency', columns='remote_ratio', values='salary_in_usd', aggfunc=sum)

    size_buble = np.array([50, 100, 150])
    min = 1000
    max = 20000
    size = np.interp(size_buble, (size_buble.min(), size_buble.max()), (min, max))
    plt.scatter(np.array(usd1.columns), np.array(usd1.iloc[0]), s=list(size), alpha=0.3)
    plt.scatter(np.array(usd2.columns), np.array(usd2.iloc[0]), s=list(size), alpha=0.3)
    plt.show()
