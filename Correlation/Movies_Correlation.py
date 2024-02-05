#importing Libraries

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

plt.style.use('ggplot')
from matplotlib.pyplot import figure

#%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8) # Adjusts the configuration of the plots we will create

# Read in the data
df = pd.read_csv(r'C:\Users\Lenovo\Desktop\DataAnalytics\Correlation\movies.csv')

#print(df.head())

#Deleting the rows for missing data
df = df.dropna()

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    #print('{} - {}%'.format(col, pct_missing))


#cheking data Types of our columns
    
#print(df.dtypes)

df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')

#print(df.dtypes)

#Sometimes released dates and year are not matched, so we need to correct them

df['year_corrected'] = df['released'].astype(str).str.extract(pat = '([0-9]{4})').astype('int64')
#print(df)

df = df.sort_values(by = ['gross'], inplace = False, ascending=False);

pd.set_option('display.max_rows' , None  )

#Drop any duplicates
df = df.drop_duplicates()

#Now finding out the correct Correlation
#1st Correlation between budget and gross

#Scatter plot with budget vs gross

plt.scatter(x = df['budget'] , y = df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Budget')    
plt.ylabel('Gross Earnings')   

#PLotting the budget vs gross using seaborn
sns.regplot(x = 'budget' , y = 'gross' , data = df , scatter_kws = {'color' : 'red'} , line_kws = {'color' : 'blue'})

df.corr(method='pearson', numeric_only=True) #Pearson, Kendall, Spearman

correlation_matrix = df.corr(method= 'pearson' , numeric_only=True)
heatmap = sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Matrix for Numeric Features')

df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

#print(df_numerized)

correlation_matrix = df_numerized.corr(method= 'pearson' , numeric_only=True)
heatmap = sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')



correlation_mat = df_numerized.corr()
corrs_pairs = correlation_mat.unstack()
sorted_pairs = corrs_pairs.sort_values()

high_corr = sorted_pairs[(sorted_pairs) > 0.5]
print(high_corr)

