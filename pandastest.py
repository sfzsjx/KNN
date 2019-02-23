import pandas as pd
import numpy as np

from pandas import Series, DataFrame
#pandas 中 DataFrame 数据结构
data = {'Chinese': [66, 95, 93, 90,80],'English': [65, 85, 92, 88, 90],'Math': [30, 98, 96, 77, 90]}
# df1= DataFrame(data)
df2 = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'], columns=['English', 'Math', 'Chinese'])
# print(df1)
print(df2)

# #数据导入和输出
# peopel: DataFrame = DataFrame(pd.read_csv('Wholesale customers data.csv'))
# peopel.to_csv('customers.csv')


#数据清洗
df2 = df2.drop(columns=['English'])
df2 = df2.drop(index=['ZhangFei'])

df2.rename(columns={'Chinese':'语文','Math':'数学'},index={'GuanYu':'关羽','ZhaoYun':'赵云','HuangZhong':'黄忠','DianWei':'典韦'},inplace=True)

df2.drop_duplicates() #去除重复行
df2['语文'].astype('int')

from pandasql import sqldf,load_meat,load_births
pysqldf = lambda sql:sqldf(sql,globals())
sql = "select * from df2 "
print(pysqldf(sql))