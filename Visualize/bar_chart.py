# Khai báo thư viện
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set(rc={"figure.figsize": (9, 6), "figure.dpi": 1000})

# Tiến hành vẽ biểu đồ
df = pd.read_csv("C:/ML/ML1/Data_Science_Salary/DATA/Data_Science_Salary.csv")

# Lọc dữ liệu cho năm 2022
df_2023 = df[df['work_year'] == 2023]

# Nhóm theo job_title và tính tổng số lần xuất hiện
job_title_counts = df_2023['job_title'].value_counts()

# Lấy Top 10 Job Titles
top_10_job_titles = job_title_counts.head(10)

# Bảng màu cho các thanh ngang
colors = plt.cm.Paired(range(len(top_10_job_titles)))

# Vẽ biểu đồ thanh ngang
plt.figure(figsize=(10, 5), dpi=100)
ax = top_10_job_titles.sort_values().plot(kind='barh', color=colors, edgecolor='black')

# Thêm chú thích số liệu trên từng thanh ngang
for i, v in enumerate(top_10_job_titles.sort_values()):
    ax.text(v + 3, i - 0.18, str(v), color='black', fontsize=14, fontweight='bold')

# Thêm lưới và điều chỉnh khoảng giá trị trục x
ax.grid(axis='x', linestyle='--', alpha=0.6)
ax.locator_params(axis='x', tight=True, nbins=25)  # Số lượng khoảng giá trị trục x

# # Giới hạn giá trị trục x đến 500
# plt.xlim(0, 500)

plt.title('Top 10 Job Titles in 2023', fontsize=22, fontweight='bold')
plt.xlabel('Number of Occurrences', fontsize=22, fontweight='bold')
plt.ylabel('Job Title', fontsize=22, fontweight='bold')

# Căn chỉnh tự động kích thước biểu đồ để nằm giữa khung hình máy
plt.tight_layout()

plt.show()