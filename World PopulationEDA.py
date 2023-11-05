#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("C:/Users/shrad/OneDrive/Desktop/plmini/world-population.csv")


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.dtypes


# In[6]:


percent_null=(df.isnull().sum())/df.count()*100


# In[7]:


percent_null


# In[8]:


df['Migrants (net)'] = df['Migrants (net)'].fillna(df['Migrants (net)'].mean())
df['MedianAge'] = df['MedianAge'].fillna(df['MedianAge'].mean())
df['Fertility Rate'] = df['Fertility Rate'].fillna(df['Fertility Rate'].mean())


# In[9]:


df.isnull().sum()


# In[10]:


df.duplicated().sum()


# In[11]:


df ['country'].value_counts()


# In[12]:


countc=df['country'].unique()


# In[13]:


countc


# In[14]:


len(countc)


# In[15]:


max_population_countries = df[df['Population'] == df['Population'].max()]
print("\nCountries with the highest population:")
print(max_population_countries[['country', 'Year', 'Population']])


# In[16]:


avg_population_change = df.groupby('Year')['Yearly  Change'].mean()
print("\nAverage change in population by year:")
print(avg_population_change)


# In[17]:


df_pop_2020 = df[df['Year'] == 2020]
min_median_age_countries = df_pop_2020[df_pop_2020['MedianAge'] == df_pop_2020['MedianAge'].min()]
print("\nCountries with lowest median age:")
print(min_median_age_countries[['country', 'Year', 'MedianAge']])


# In[18]:


max_fertility_rate_countries = df[df['Fertility Rate'] == df['Fertility Rate'].max()]
print("\nCountries with highest fertility rate:")
print(max_fertility_rate_countries[['country', 'Year', 'Fertility Rate']])


# In[19]:


df_pop_2020 = df[df['Year'] == 2020]
top_10_populated_countries = df_pop_2020.nlargest(10, 'Population')

plt.bar(top_10_populated_countries['country'], top_10_populated_countries['Population'])
plt.xlabel('Country')
plt.ylabel('Population')
plt.title('Top 10 largest coutries')
plt.xticks(rotation=90)
plt.show()


# In[20]:


df['Year'].value_counts()


# In[21]:


condition =df['country']=='China'


# In[22]:


country=df[condition].country
years=df[condition].Year
population=df[condition].Population


# In[23]:


plt.plot(years,population)
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("China Population")
plt.legend(['China poulation'])


# In[24]:


condition1 =df['country']=='India'


# In[25]:


country=df[condition1].country
years=df[condition1].Year
population=df[condition1].Population


# In[26]:


plt.plot(years,population)
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("India Population")
plt.legend(['India poulation'])


# In[27]:


conditionafghanistan =df['country']=='Afghanistan'
conditionangola =df['country']=='Angola'

chn=df[condition].country
ind=df[condition1].country
afg=df[conditionafghanistan].country
ang=df[conditionangola].country

popc=df[condition].Population
popi=df[condition1].Population
popa=df[conditionafghanistan].Population
popan=df[conditionangola].Population

agec=df[condition].MedianAge
agei=df[condition1].MedianAge
agea=df[conditionafghanistan].MedianAge
agean=df[conditionangola].MedianAge

plt.plot(popc,agec)
plt.plot(popi,agei)
plt.plot(popa,agea)
plt.plot(popan,agean)
plt.xlabel("Population")
plt.ylabel("MedianAge")
plt.title("Median age distribution")
plt.legend(['China','India','Afghanistan','Angola'])
plt.show()


# In[28]:


yearsc=df[condition].Year
yearsi=df[condition1].Year
yearsa=df[conditionafghanistan].Year
yearsang=df[conditionangola].Year
rankc=df[condition].Rank
ranki=df[condition1].Rank
ranka=df[conditionafghanistan].Rank
rankang=df[conditionangola].Rank


# In[29]:


plt.plot(yearsc,rankc)
plt.plot(yearsi,ranki)
plt.plot(yearsa,ranka)
plt.plot(yearsang,rankang)
plt.xlabel("Year")
plt.ylabel("Rank")
plt.legend(['China','India','Afghanistan','Angola'])


# In[30]:


plt.plot(yearsc,popc)
plt.plot(yearsi,popi)
plt.xlabel("Year")
plt.ylabel("Pouplation")
plt.legend(['China','India'])


# In[31]:


country = df.groupby('country')
country.head(5)


# In[32]:


india_data=country.get_group('India')
india_data


# In[33]:


Year = df.groupby('Year')
Year.first()

Year_2020  = Year.get_group(2020)
Year_2020


# In[34]:


plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
plt.plot(india_data['Year'],india_data['Migrants (net)'])
plt.xlabel('Year')
plt.ylabel('Migrants')
plt.title('Migrants in India')
plt.subplot(2,2,2)
plt.plot(india_data['Year'],india_data['MedianAge'])
plt.xlabel('Year')
plt.ylabel('MedianAge')
plt.title('Median age of population in India')
plt.subplot(2,2,3)
plt.plot(india_data['Year'],india_data['Fertility Rate'])
plt.xlabel('Year')
plt.ylabel('Fertility Rate')
plt.title('Fertility Rate')
plt.subplot(2,2,4)
plt.plot(india_data['Year'],india_data['Density (P/Km²)'])
plt.xlabel('Year')
plt.ylabel('Density (P/Km²)')
plt.title('Density (P/Km²)')


# In[35]:


def converter(y, pos):
    return '{:.1f} Billion'.format(y * 1e-9)


# In[36]:


plt.figure(figsize=(12,5))

plt.plot(india_data['Year'],india_data['World Population'])

plt.xlabel('Year')
plt.ylabel('Population')
plt.title('India')

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(india_data['Year'],india_data['World Population'])

ax.yaxis.set_major_formatter(converter)


ax.set_xlabel('Year')
ax.set_ylabel('Population')
ax.set_title('India')


# In[37]:


us_data=country.get_group('United States')
us_data


# In[38]:


plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
plt.plot(us_data['Year'],us_data['Migrants (net)'])
plt.xlabel('Year')
plt.ylabel('Migrants')
plt.title('Migrants in United States')
plt.subplot(2,2,2)
plt.plot(us_data['Year'],us_data['MedianAge'])
plt.xlabel('Year')
plt.ylabel('Median Age')
plt.title('Median age of population in United States')
plt.subplot(2,2,3)
plt.plot(us_data['Year'],us_data['Fertility Rate'])
plt.xlabel('Year')
plt.ylabel('Fertility Rate')
plt.title('Fertility Rate')
plt.subplot(2,2,4)
plt.plot(us_data['Year'],us_data['Density (P/Km²)'])
plt.xlabel('Year')
plt.ylabel('Density (P/Km²)')
plt.title('Density (P/Km²)')


# Population Rank Change of India and Neighbouring Countries¶

# In[39]:


china_data=country.get_group('China')
china_data


# In[40]:


plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
plt.plot(china_data['Year'],china_data['Migrants (net)'])
plt.xlabel('Year')
plt.ylabel('Migrants')
plt.title('Migrants in China')
plt.subplot(2,2,2)
plt.plot(china_data['Year'],china_data['MedianAge'])
plt.xlabel('Year')
plt.ylabel('Median Age')
plt.title('Median age of population in China')
plt.subplot(2,2,3)
plt.plot(china_data['Year'],china_data['Fertility Rate'])
plt.xlabel('Year')
plt.ylabel('Fertility Rate')
plt.title('Fertility Rate')
plt.subplot(2,2,4)
plt.plot(china_data['Year'],china_data['Density (P/Km²)'])
plt.xlabel('Year')
plt.ylabel('Density (P/Km²)')
plt.title('Density (P/Km²)')


# In[41]:


indonesia_data=country.get_group('Indonesia')
indonesia_data


# In[42]:


plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
plt.plot(indonesia_data['Year'],indonesia_data['Migrants (net)'])
plt.xlabel('Year')
plt.ylabel('Migrants')
plt.title('Migrants in indonesia')
plt.subplot(2,2,2)
plt.plot(indonesia_data['Year'],indonesia_data['MedianAge'])
plt.xlabel('Year')
plt.ylabel('Median Age')
plt.title('Median age of population in indonesia')
plt.subplot(2,2,3)
plt.plot(indonesia_data['Year'],indonesia_data['Fertility Rate'])
plt.xlabel('Year')
plt.ylabel('Fertility Rate')
plt.title('Fertility Rate')
plt.subplot(2,2,4)
plt.plot(indonesia_data['Year'],indonesia_data['Density (P/Km²)'])
plt.xlabel('Year')
plt.ylabel('Density (P/Km²)')
plt.title('Density (P/Km²)')


# In[43]:


brazil_data=country.get_group('Brazil')
brazil_data.head()


# In[44]:


plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
plt.plot(brazil_data['Year'],brazil_data['Migrants (net)'])
plt.xlabel('Year')
plt.ylabel('Migrants')
plt.title('Migrants in brazil')
plt.subplot(2,2,2)
plt.plot(brazil_data['Year'],brazil_data['MedianAge'])
plt.xlabel('Year')
plt.ylabel('Median Age')
plt.title('Median age of population in brazil')
plt.subplot(2,2,3)
plt.plot(brazil_data['Year'],brazil_data['Fertility Rate'])
plt.xlabel('Year')
plt.ylabel('Fertility Rate')
plt.title('Fertility Rate')
plt.subplot(2,2,4)
plt.plot(brazil_data['Year'],brazil_data['Density (P/Km²)'])
plt.xlabel('Year')
plt.ylabel('Density (P/Km²)')
plt.title('Density (P/Km²)')


# In[45]:


pakistan_data=country.get_group('Pakistan')
pakistan_data.head()


# In[46]:


plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
plt.plot(pakistan_data['Year'],pakistan_data['Migrants (net)'])
plt.xlabel('Year')
plt.ylabel('Migrants')
plt.title('Migrants in Pakistan')
plt.subplot(2,2,2)
plt.plot(pakistan_data['Year'],pakistan_data['MedianAge'])
plt.xlabel('Year')
plt.ylabel('Median Age')
plt.title('Median age of population in Pakistan')
plt.subplot(2,2,3)
plt.plot(pakistan_data['Year'],pakistan_data['Fertility Rate'])
plt.xlabel('Year')
plt.ylabel('Fertility Rate')
plt.title('Fertility Rate')
plt.subplot(2,2,4)
plt.plot(pakistan_data['Year'],pakistan_data['Density (P/Km²)'])
plt.xlabel('Year')
plt.ylabel('Density (P/Km²)')
plt.title('Density (P/Km²)')


# In[47]:


maldives_data=country.get_group('Maldives')
maldives_data.head()


# In[48]:


plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
plt.plot(maldives_data['Year'],maldives_data['Migrants (net)'])
plt.xlabel('Year')
plt.ylabel('Migrants')
plt.title('Migrants in Maldives')
plt.subplot(2,2,2)
plt.plot(maldives_data['Year'],maldives_data['MedianAge'])
plt.xlabel('Year')
plt.ylabel('Median Age')
plt.title('Median age of population in maldives')
plt.subplot(2,2,3)
plt.plot(maldives_data['Year'],maldives_data['Fertility Rate'])
plt.xlabel('Year')
plt.ylabel('Fertility Rate')
plt.title('Fertility Rate')
plt.subplot(2,2,4)
plt.plot(maldives_data['Year'],maldives_data['Density (P/Km²)'])
plt.xlabel('Year')
plt.ylabel('Density (P/Km²)')
plt.title('Density (P/Km²)')


# In[49]:


nepal_data=country.get_group('Nepal')
nepal_data.head()


# In[50]:


plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
plt.plot(nepal_data['Year'],nepal_data['Migrants (net)'])
plt.xlabel('Year')
plt.ylabel('Migrants')
plt.title('Migrants in Nepal')
plt.subplot(2,2,2)
plt.plot(nepal_data['Year'],nepal_data['MedianAge'])
plt.xlabel('Year')
plt.ylabel('Median Age')
plt.title('Median age of population in Nepal')
plt.subplot(2,2,3)
plt.plot(nepal_data['Year'],nepal_data['Fertility Rate'])
plt.xlabel('Year')
plt.ylabel('Fertility Rate')
plt.title('Fertility Rate')
plt.subplot(2,2,4)
plt.plot(nepal_data['Year'],nepal_data['Density (P/Km²)'])
plt.xlabel('Year')
plt.ylabel('Density (P/Km²)')
plt.title('Density (P/Km²)')


# In[51]:


df_sorted = df.sort_values(['Year', 'Rank'])
countries = ['India', 'Afghanistan', 'Bangladesh', 'Maldives', 'Pakistan', 'Myanmar', 'Sri Lanka','China','Nepal']
plt.figure(figsize=(10, 6))
for country in countries:
    country_data = df_sorted[df_sorted['country'] == country]
    plt.plot(country_data['Year'], country_data['Rank'], label=country)

plt.title('Rank of Countries Over Time')
plt.xlabel('Year')
plt.ylabel('Rank')

plt.legend(title='Country')

plt.show()


# In[52]:


rank_data = df.groupby(['country', 'Year'])['Rank'].min().unstack()

# Create the plot
rank_data.plot(marker='o', figsize=(10, 6))

# Set the plot title and axis labels
plt.title('Rank of Countries Year-wise')
plt.xlabel('Year')
plt.ylabel('Rank')

# Add a legend
plt.legend(title='Country')

# Display the plot
plt.show()


# In[53]:


plt.figure(figsize=(45,20))
sns.lineplot(data=df,x='country',y='Rank')
plt.xticks(rotation=90);


# In[54]:


plt.figure(figsize=(12,11))
sns.barplot(data=country_data,x='Year',y="Population",hue='MedianAge',width=2)
plt.xticks(rotation=90);


# In[55]:


df_world = df[['Year', 'World Population']].groupby('Year')['World Population'].mean()
df_world


# In[56]:


def converter(y, pos):
    return '{:.1f} Billion'.format(y * 1e-9)


# In[57]:


fig, ax = plt.subplots(figsize=(20,6))

ax.plot(df_world.index, df_world.values)
ax.plot(df_world.index, df_world.values, 'o', color = 'r')

ax.yaxis.set_major_formatter(converter)

ax.set_xticks(df_world.index)

for ticks in ax.get_xticklabels():
    ticks.set_rotation(45)
    ticks.set_horizontalalignment('center')

ax.grid(True, alpha = 0.5)

ax.set_xlabel('Year')
ax.set_ylabel('Population')
ax.set_title('World Population')


# Population Study in Indian Subcontinent¶

# In[58]:


indian_sub = df[df['country'].isin(countries)]


# In[59]:


indian_sub.head()


# In[60]:


indian_sub_group = indian_sub.groupby(['country', 'Year'])['Population', 'Yearly %   Change', 'Yearly  Change',
                                        'Migrants (net)', 'MedianAge', 'Fertility Rate', 'Density (P/Km²)'].mean().reset_index()


# In[61]:


indian_sub_group.head()


# In[62]:


for i in indian_sub_group['country'].unique():
    print(i)


# In[63]:


plt.figure(figsize=(100,10))
fig, ax = plt.subplots()
for i in indian_sub_group['country'].unique():
    ax.plot(indian_sub_group[indian_sub_group.country == i]['Year'], indian_sub_group[indian_sub_group.country == i]['Population'],'-o', label = i)

ax.legend()
ax.set_xticks(indian_sub_group['Year'].unique())

for ticks in ax.get_xticklabels():
    ticks.set_rotation(90)
    
ax.set_xlabel('Year')
ax.set_ylabel('Population')
ax.yaxis.set_major_formatter(converter)

fig.show()


# In[64]:


import plotly.express as px


# In[65]:


pop_density_after_2015 = indian_sub_group.query("Year > 2015")
pop_density_after_2015.head()


# In[66]:


fig = px.sunburst(
            pop_density_after_2015,
            path = ['country', 'Year'],
            values = 'Density (P/Km²)'
                 )

fig.update_layout(title_text = 'Population Density After 2015', title_x = 0.5)


# In[67]:


fig = px.sunburst(
            pop_density_after_2015,
            path = ['country', 'Year'],
            values = 'Countrys Share of  World Pop'
                 )

fig.update_layout(title_text = 'Population Density After 2015', title_x = 0.5)


# In[ ]:


sns.scatterplot(x =df ['MedianAge'], y = df['Fertility Rate'])


# In[68]:


sns.lineplot(x = df['Migrants (net)'], y = df['Fertility Rate'])


# In[69]:


fig = px.sunburst(
            pop_density_after_2015,
            path = ['country', 'Year'],
            values = 'Fertility Rate'
                 )

fig.update_layout(title_text = 'Fertility Rate After 2015', title_x = 0.5)


# In[ ]:




