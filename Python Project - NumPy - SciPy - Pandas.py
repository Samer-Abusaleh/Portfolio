
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
from pprint import pprint
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison


# # Category files

# In[2]:


def get_category(json_file):
    category_dict = defaultdict(int)
    with open (json_file) as file:
        response = json.load(file)
        for item in response['items']:
            category_dict[item['id']] = item['snippet']['title']
        return category_dict
category_us = get_category("US_category_id.json")
category_gb = get_category("GB_category_id.json")
category_ca = get_category("CA_category_id.json")
category_de = get_category("DE_category_id.json")
category_fr = get_category("FR_category_id.json")


# # Data: Function Top50

# In[3]:


#Read CSV
us = pd.read_csv("USvideos.csv")
gb = pd.read_csv("GBvideos.csv")
ca = pd.read_csv("CAvideos.csv")
de = pd.read_csv("DEvideos.csv")
fr = pd.read_csv("FRvideos.csv")

#Function for converting CSV into Top50 by month
def raw_to_top50(df, category_file):
    df['trending_date'] = pd.to_datetime(df['trending_date'], format = '%y.%d.%m')
    df['trending_YearMonth'] = df['trending_date'].apply(lambda x: f'{x.year}-{x.month}')
    df['publish_time'] = pd.to_datetime(df['publish_time'], format = '%Y-%m-%dT%H:%M:%S.%fZ')
    count_byMonth = df.groupby(['trending_YearMonth','title','category_id'])['video_id','views'].count() 
    top50_byMonth = count_byMonth['video_id'].groupby(level=0, group_keys=False).nlargest(50)
    top50_byMonth = pd.DataFrame(top50_byMonth)
    top50_byMonth = top50_byMonth.rename(columns={'video_id':'video_id_count'})
    top50_byMonth = top50_byMonth.reset_index()
    top50_byMonth['category_id'] = top50_byMonth['category_id'].astype(str)
    top50_byMonth['category'] = top50_byMonth['category_id'].map(lambda x: category_file.get(x))
    return top50_byMonth


# In[4]:


us_top50_byMonth = raw_to_top50(us, category_us)
us_top50_byMonth.head()


# In[5]:


gb_top50_byMonth = raw_to_top50(gb, category_gb)
gb_top50_byMonth.head()


# In[6]:


ca_top50_byMonth = raw_to_top50(ca, category_ca)
ca_top50_byMonth.head()


# In[7]:


de_top50_byMonth = raw_to_top50(de, category_de)
de_top50_byMonth.head()


# In[8]:


fr_top50_byMonth = raw_to_top50(fr, category_fr)
fr_top50_byMonth.head()


# # Count of Trending Days: Bar Charts

# In[9]:


months = ['2017-11','2017-12','2018-1','2018-2','2018-3','2018-4','2018-5','2018-6']
for month in months:
    us = us_top50_byMonth.loc[us_top50_byMonth['trending_YearMonth']==month,:]
    gb = gb_top50_byMonth.loc[gb_top50_byMonth['trending_YearMonth']==month,:]
    ca = ca_top50_byMonth.loc[ca_top50_byMonth['trending_YearMonth']==month,:]
    de = de_top50_byMonth.loc[de_top50_byMonth['trending_YearMonth']==month,:]
    fr = fr_top50_byMonth.loc[fr_top50_byMonth['trending_YearMonth']==month,:]
    plt.figure(figsize=(20,25))
    plt.suptitle(f'Number of Trending Days in {month}', fontsize=20)
    plt.subplot(5,1,1,)
    plt.bar(np.arange(50),us['video_id_count'])
    plt.title('US', fontsize=14)
    plt.xticks(np.arange(50),np.arange(1,51,1))
    plt.ylim(0,31)
    plt.yticks(np.arange(0,35,5),[0,5,10,15,20,25,30])
    plt.ylabel('Number Trending Days', fontsize=14)
    plt.subplot(5,1,2)
    plt.bar(np.arange(50),gb['video_id_count'])
    plt.title('GB', fontsize=14)
    plt.xticks(np.arange(50),np.arange(1,51,1))
    plt.ylim(0,31)
    plt.yticks(np.arange(0,35,5),[0,5,10,15,20,25,30])
    plt.ylabel('Number Trending Days', fontsize=14)
    plt.subplot(5,1,3)  
    plt.bar(np.arange(50),ca['video_id_count'])
    plt.title('CA', fontsize=14)
    plt.xticks(np.arange(50),np.arange(1,51,1))
    plt.ylim(0,31)
    plt.yticks(np.arange(0,35,5),[0,5,10,15,20,25,30])
    plt.ylabel('Number Trending Days', fontsize=14)
    plt.subplot(5,1,4)   
    plt.bar(np.arange(50),de['video_id_count'])
    plt.title('DE', fontsize=14)
    plt.xticks(np.arange(50),np.arange(1,51,1))
    plt.ylim(0,31)
    plt.yticks(np.arange(0,35,5),[0,5,10,15,20,25,30])
    plt.ylabel('Number Trending Days', fontsize=14)
    plt.subplot(5,1,5)    
    plt.bar(np.arange(50),fr['video_id_count'])
    plt.xticks(np.arange(50),np.arange(1,51,1))
    plt.ylim(0,31)
    plt.yticks(np.arange(0,35,5),[0,5,10,15,20,25,30])
    plt.title('FR', fontsize=14)
    plt.ylabel('Number Trending Days', fontsize=14)
    plt.savefig('Views_{month}.png')


# In[10]:


# Average number of trending days of top 50 videos across months & countries
ca_AvgNTDByMonth = ca_top50_byMonth.groupby(['trending_YearMonth'])['video_id_count'].mean()
plt.bar(ca_top50_byMonth['trending_YearMonth'],ca_top50_byMonth['video_id_count'])
plt.ylim(0,31)
plt.yticks(np.arange(0,35,5),[0,5,10,15,20,25,30])
plt.title('Avg num of trending days of top 50 videos: CA', fontsize=14)
plt.show()
de_AvgNTDByMonth = de_top50_byMonth.groupby(['trending_YearMonth'])['video_id_count'].mean()
plt.bar(de_top50_byMonth['trending_YearMonth'],de_top50_byMonth['video_id_count'])
plt.ylim(0,31)
plt.yticks(np.arange(0,35,5),[0,5,10,15,20,25,30])
plt.title('Avg num of trending days of top 50 videos: DE', fontsize=14)
plt.show()
fr_AvgNTDByMonth = fr_top50_byMonth.groupby(['trending_YearMonth'])['video_id_count'].mean()
plt.bar(fr_top50_byMonth['trending_YearMonth'],fr_top50_byMonth['video_id_count'])
plt.ylim(0,31)
plt.yticks(np.arange(0,35,5),[0,5,10,15,20,25,30])
plt.title('Avg num of trending days of top 50 videos: FR', fontsize=14)
plt.show()
gb_AvgNTDByMonth = gb_top50_byMonth.groupby(['trending_YearMonth'])['video_id_count'].mean()
plt.bar(gb_top50_byMonth['trending_YearMonth'],gb_top50_byMonth['video_id_count'])
plt.ylim(0,31)
plt.yticks(np.arange(0,35,5),[0,5,10,15,20,25,30])
plt.title('Avg num of trending days of top 50 videos: GB', fontsize=14)
plt.show()
us_AvgNTDByMonth = us_top50_byMonth.groupby(['trending_YearMonth'])['video_id_count'].mean()
plt.bar(us_top50_byMonth['trending_YearMonth'],us_top50_byMonth['video_id_count'])
plt.ylim(0,31)
plt.yticks(np.arange(0,35,5),[0,5,10,15,20,25,30])
plt.title('Avg num of trending days of top 50 videos: US', fontsize=14)
plt.show()


# # Category Overlap across countries

# In[11]:


for month in months:
    us_cat = us_top50_byMonth.loc[us_top50_byMonth['trending_YearMonth'] == month]['category'].value_counts()
    gb_cat = gb_top50_byMonth.loc[gb_top50_byMonth['trending_YearMonth'] == month]['category'].value_counts()
    ca_cat = ca_top50_byMonth.loc[ca_top50_byMonth['trending_YearMonth'] == month]['category'].value_counts()
    de_cat = de_top50_byMonth.loc[de_top50_byMonth['trending_YearMonth'] == month]['category'].value_counts()
    fr_cat = fr_top50_byMonth.loc[fr_top50_byMonth['trending_YearMonth'] == month]['category'].value_counts()
    
    # plot pie charts
    plt.figure(figsize=(20,10))
    plt.suptitle(f'{month}')
    plt.subplot(2,3,1)
    plt.pie(us_cat,labels = us_cat.index, explode = [.05 for x in np.arange(len(us_cat))], autopct = '%.0f%%', startangle = 45)
    plt.axis('equal')
    plt.title('US')
    plt.subplot(2,3,2)
    plt.pie(gb_cat,labels = gb_cat.index, explode = [.05 for x in np.arange(len(gb_cat))], autopct = '%.0f%%', startangle = 45)
    plt.axis('equal')
    plt.title('GB')
    plt.subplot(2,3,3)
    plt.pie(ca_cat,labels = ca_cat.index, explode = [.05 for x in np.arange(len(ca_cat))], autopct = '%.0f%%', startangle = 45)
    plt.axis('equal')
    plt.title('CA')
    plt.subplot(2,3,4)
    plt.pie(de_cat,labels = de_cat.index, explode = [.05 for x in np.arange(len(de_cat))], autopct = '%.0f%%', startangle = 45)
    plt.axis('equal')
    plt.title('DE')
    plt.subplot(2,3,5)
    plt.pie(fr_cat,labels = fr_cat.index, explode = [.05 for x in np.arange(len(fr_cat))], autopct = '%.0f%%', startangle = 45)
    plt.axis('equal')
    plt.title('FR')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=None)
    plt.savefig(f"Pie_{month}.png")
    
    # plot stacked bar charts
    unique_cats = us_cat.index.append(gb_cat.index).append(ca_cat.index).append(de_cat.index).append(fr_cat.index).unique()
    cats = {}
    for key in unique_cats:
        cats[key]=0
    us_dict = us_cat.to_dict()
    us_cats = cats.copy()
    for item in us_dict:
        us_cats[item]=us_cats[item]+us_dict[item]

    gb_dict = gb_cat.to_dict()
    gb_cats = cats.copy()
    for item in gb_dict:
        gb_cats[item]=gb_cats[item]+gb_dict[item] 

    ca_dict = ca_cat.to_dict()
    ca_cats = cats.copy()
    for item in ca_dict:
        ca_cats[item]=ca_cats[item]+ca_dict[item] 

    de_dict = de_cat.to_dict()
    de_cats = cats.copy()
    for item in de_dict:
        de_cats[item]=de_cats[item]+de_dict[item] 
        
    fr_dict = fr_cat.to_dict()
    fr_cats = cats.copy()
    for item in fr_dict:
        fr_cats[item]=fr_cats[item]+fr_dict[item] 
    
    plt.figure(figsize = (25,6))
    p1= plt.bar(np.arange(len(us_cats.keys())),us_cats.values())
    plt.ylabel("Number of Videos", fontsize = 12)
    plt.xticks(np.arange(len(us_cats.keys())),list(us_cats.keys()),fontsize = 11)
    plt.title(f"Overlap in Categories of Top-20 Videos across Countries {month}", fontsize = 14)
    bottom_us = list(us_cats.values())
    p2 = plt.bar(np.arange(len(gb_cats.keys())),gb_cats.values(), bottom=bottom_us)
    bottom_gb = list(gb_cats.values())
    bottom_us_gb = [sum(x) for x in zip(bottom_us,bottom_gb)]
    p3 = plt.bar(np.arange(len(ca_cats.keys())),ca_cats.values(), bottom=bottom_us_gb)
    bottom_ca = list(ca_cats.values())
    bottom_us_gb_ca = [sum(x) for x in zip(bottom_us_gb,bottom_ca)]
    p4 = plt.bar(np.arange(len(de_cats.keys())),de_cats.values(), bottom=bottom_us_gb_ca)
    bottom_de = list(de_cats.values())
    bottom_us_gb_ca_de = [sum(x) for x in zip(bottom_us_gb_ca,bottom_de)]
    p5 = plt.bar(np.arange(len(fr_cats.keys())),fr_cats.values(), bottom=bottom_us_gb_ca_de)
    plt.legend((p1,p2,p3,p4,p5),("US",'GB','CA','DE','FR'))
    plt.savefig(f"stacked_bar_{month}.png")


# # Video overlap rates across countries

# In[12]:


us_gb=[]
us_ca=[]
us_de=[]
us_fr=[]
gb_ca=[]
gb_de=[]
gb_fr=[]
ca_de=[]
ca_fr=[]
de_fr=[]

for month in months:
    us_title = us_top50_byMonth.loc[us_top50_byMonth['trending_YearMonth'] == month]['title']
    gb_title = gb_top50_byMonth.loc[gb_top50_byMonth['trending_YearMonth'] == month]['title']
    ca_title = ca_top50_byMonth.loc[ca_top50_byMonth['trending_YearMonth'] == month]['title']
    de_title = de_top50_byMonth.loc[de_top50_byMonth['trending_YearMonth'] == month]['title']
    fr_title = fr_top50_byMonth.loc[fr_top50_byMonth['trending_YearMonth'] == month]['title']

    count = 0
    for item in gb_title:
        if item in list(us_title):
            count = count + 1
    us_gb_overlap = int(count/50*100)
    
    
    count = 0
    for item in ca_title:
        if item in list(us_title):
            count = count + 1
    us_ca_overlap = int(count/50*100)
 
    
    count = 0
    for item in de_title:
        if item in list(us_title):
            count = count + 1
    us_de_overlap = int(count/50*100)

    count = 0
    for item in fr_title:
        if item in list(us_title):
            count = count + 1
    us_fr_overlap = int(count/50*100)

    count = 0
    for item in ca_title:
        if item in list(gb_title):
            count = count + 1
    gb_ca_overlap = int(count/50*100)

    count = 0
    for item in de_title:
        if item in list(gb_title):
            count = count + 1
    gb_de_overlap = int(count/50*100)
    
    count = 0
    for item in fr_title:
        if item in list(gb_title):
            count = count + 1
    gb_fr_overlap = int(count/50*100)

    count = 0
    for item in de_title:
        if item in list(ca_title):
            count = count + 1
    ca_de_overlap = int(count/50*100)
    
    count = 0
    for item in fr_title:
        if item in list(ca_title):
            count = count + 1
    ca_fr_overlap = int(count/50*100)
    
    count = 0
    for item in fr_title:
        if item in list(de_title):
            count = count + 1
    de_fr_overlap = int(count/50*100)
        
    
    countries = ['US','GB','CA','DE','FR']
    us_overlaps = [us_gb_overlap,us_ca_overlap,us_de_overlap,us_fr_overlap]
    us_mean = np.mean(us_overlaps)
    us_yrr = stats.sem(us_overlaps)
    gb_overlaps = [us_gb_overlap,gb_ca_overlap,gb_de_overlap,gb_fr_overlap]
    gb_mean = np.mean(gb_overlaps)
    gb_yrr = stats.sem(gb_overlaps)
    ca_overlaps = [us_ca_overlap,gb_ca_overlap,ca_de_overlap,ca_fr_overlap]
    ca_mean = np.mean(ca_overlaps)
    ca_yrr = stats.sem(ca_overlaps)
    de_overlaps = [us_de_overlap,gb_de_overlap,ca_de_overlap,de_fr_overlap]
    de_mean = np.mean(de_overlaps)
    de_yrr = stats.sem(de_overlaps)
    fr_overlaps = [us_fr_overlap,gb_fr_overlap,ca_fr_overlap,de_fr_overlap]
    fr_mean = np.mean(fr_overlaps)
    fr_yrr = stats.sem(fr_overlaps)
    all_yrr = [us_yrr, gb_yrr, ca_yrr, de_yrr, fr_yrr]
    
    plt.figure(figsize = (18,5))
    plt.suptitle(f'{month}', fontsize = 16)
    plt.subplot(1,5,1)
    plt.bar(x=np.arange(4) ,height=us_overlaps, color='blue')
    plt.xticks(np.arange(4),['GB','CA','DE','FR'])
    plt.ylim(0,50)
    plt.ylabel("Percent Overlap of Top 50 Videos")
    plt.title("US")
    plt.subplot(1,5,2)
    plt.bar(x=np.arange(4),height=gb_overlaps, color='blue')
    plt.xticks(np.arange(4),['US','CA','DE','FR'])
    plt.title("GB")
    plt.ylim(0,50)
    plt.subplot(1,5,3)
    plt.bar(x=np.arange(4),height=ca_overlaps,color='blue')
    plt.xticks(np.arange(4),['US','GB','DE','FR'])
    plt.title("CA")
    plt.ylim(0,50)
    plt.subplot(1,5,4)
    plt.bar(x=np.arange(4),height=de_overlaps, color='blue')
    plt.xticks(np.arange(4),['US','GB','CA','FR'])
    plt.title("DE")
    plt.ylim(0,50)
    plt.subplot(1,5,5)
    plt.bar(x=np.arange(4),height=fr_overlaps, color='blue')
    plt.xticks(np.arange(4),['US','GB','CA','DE'])
    plt.title("FR")
    plt.ylim(0,50)
    plt.savefig(f'Overlap_rate_countries_{month}.png')
    plt.show()
    
    plt.bar(np.arange(5),height=[us_mean,gb_mean,ca_mean,de_mean,fr_mean],yerr = all_yrr, color='purple')
    plt.xticks(np.arange(5),['US','GB','CA','DE','FR'])
    plt.ylabel("Average Percent Overlap")
    plt.title(f'{month}')
    plt.ylim(0,50)
    plt.annotate(f'{us_mean:.1f}',xy=(0-.15,us_mean +6))
    plt.annotate(f'{gb_mean:.1f}',xy=(1-.15,gb_mean +6))
    plt.annotate(f'{ca_mean:.1f}',xy=(2-.15,ca_mean +6))
    plt.annotate(f'{de_mean:.1f}',xy=(3-.15,de_mean +6))
    plt.annotate(f'{fr_mean:.1f}',xy=(4-.15,fr_mean +6))
    plt.savefig(f'Overlap_rate_{month}.png')
    
    
    anova_test = stats.f_oneway(us_overlaps, gb_overlaps, ca_overlaps, de_overlaps, fr_overlaps)
    p_value = list(anova_test)[1]
    if p_value <= .05:
        overlap = us_overlaps + gb_overlaps + ca_overlaps + de_overlaps + fr_overlaps
        country = ['us']*4+['gb']*4+['ca']*4+['de']*4+['fr']*4
        df = pd.DataFrame({'Overlap': overlap, 
                          'Countries': country})
        mc = MultiComparison(df['Overlap'],df['Countries'])
        result = mc.tukeyhsd()
        print(f'Post Hoc Analysis for {month}: {result}')
    else:
        print('No significant difference across the countries in overlap rates for {month}')
    #get lists for paired-country analysis for all the months
    
    us_gb.append(us_gb_overlap)
    us_ca.append(us_ca_overlap)
    us_de.append(us_de_overlap)
    us_fr.append(us_fr_overlap)
    gb_ca.append(gb_ca_overlap)
    gb_de.append(gb_ca_overlap)
    gb_fr.append(gb_fr_overlap)
    ca_de.append(ca_de_overlap)
    ca_fr.append(ca_fr_overlap)
    de_fr.append(de_fr_overlap)


# # Video overlap rates for country pairs

# In[13]:


#make a df for repeated ANOVA
rates = us_gb + us_ca + us_de + us_fr + gb_ca + gb_de + gb_fr + ca_de + ca_fr + de_fr
across_factor = np.repeat(['us_gb','us_ca','us_de','us_fr','gb_ca','gb_de','gb_fr','ca_de','ca_fr','de_fr'],8)
within_factor = [1,2,3,4,5,6,7,8]*10
repeated = pd.DataFrame({
    "Rates": rates,
    "Across_factor": across_factor,
    "Within_factor": within_factor
})


print(us_gb)
print(us_ca)
print(us_de)
print(us_fr)
print(gb_ca)
print(gb_de)
print(gb_fr)
print(ca_de)
print(ca_fr)
print(de_fr)
print(stats.f_oneway(us_gb,us_ca,us_de,us_fr,gb_ca,gb_de,gb_fr,ca_de,ca_fr,de_fr))
mc = MultiComparison(repeated['Rates'],repeated['Across_factor'])
result = mc.tukeyhsd()
print(f'posthoc results: {result}')

plt.figure(figsize=(30,25))
plt.subplot(2,5,1)
plt.bar(np.arange(8),us_gb)
plt.xticks(np.arange(8),months, rotation = 60, ha='right', fontsize=14)
plt.yticks(np.arange(0,101,20), fontsize = 14)
plt.title('US_GB', fontsize = 24)
plt.ylabel('Overlap Rates (percent)', fontsize =20)
plt.ylim(0,50)
plt.subplot(2,5,2)
plt.bar(np.arange(8),us_ca)
plt.xticks(np.arange(8),months, rotation = 60, ha='right', fontsize=14)
plt.yticks(np.arange(0,101,20), fontsize = 14)
plt.title('US_CA', fontsize = 24)
plt.ylim(0,50)
plt.subplot(2,5,3)
plt.bar(np.arange(8),us_de)
plt.xticks(np.arange(8),months, rotation = 60, ha='right', fontsize=14)
plt.yticks(np.arange(0,101,20), fontsize = 14)
plt.title('US_DE', fontsize = 24)
plt.ylim(0,50)
plt.subplot(2,5,4)
plt.bar(np.arange(8),us_fr)
plt.xticks(np.arange(8),months, rotation = 60, ha='right', fontsize=14)
plt.yticks(np.arange(0,101,20), fontsize = 14)
plt.yticks(np.arange(0,101,20), fontsize = 14)
plt.title('US_FR', fontsize = 24)
plt.ylim(0,50)
plt.subplot(2,5,5)
plt.bar(np.arange(8),gb_ca)
plt.xticks(np.arange(8),months, rotation = 60, ha='right', fontsize=14)
plt.yticks(np.arange(0,101,20), fontsize = 14)
plt.title('GB_CA', fontsize = 24)
plt.ylim(0,50)
plt.subplot(2,5,6)
plt.bar(np.arange(8),gb_de)
plt.xticks(np.arange(8),months, rotation = 60, ha='right', fontsize=14)
plt.yticks(np.arange(0,102,20), fontsize = 14)
plt.ylabel('Overlap Rates (percent)', fontsize =20)
plt.title('GB_DE', fontsize = 24)
plt.ylim(0,50)
plt.subplot(2,5,7)
plt.bar(np.arange(8),gb_fr)
plt.xticks(np.arange(8),months, rotation = 60, ha='right', fontsize=14)
plt.yticks(np.arange(0,101,20), fontsize = 14)
plt.title('GB_FR', fontsize = 24)
plt.ylim(0,50)
plt.subplot(2,5,8)
plt.bar(np.arange(8),ca_de)
plt.xticks(np.arange(8),months, rotation = 60, ha='right', fontsize=14)
plt.yticks(np.arange(0,101,20), fontsize = 14)
plt.title('CA_DE', fontsize = 24)
plt.ylim(0,50)
plt.subplot(2,5,9)
plt.bar(np.arange(8),ca_fr)
plt.xticks(np.arange(8),months, rotation = 60, ha='right', fontsize=14)
plt.yticks(np.arange(0,101,20), fontsize = 14)
plt.title('CA_FR', fontsize = 24)
plt.ylim(0,50)
plt.subplot(2,5,10)
plt.bar(np.arange(8),de_fr)
plt.xticks(np.arange(8),months, rotation = 60, ha='right', fontsize=14)
plt.yticks(np.arange(0,101,20), fontsize = 14)
plt.title('DE_FR', fontsize = 24)
plt.ylim(0,50)
plt.savefig("Overlap_country_pairs.png")


# In[14]:


repeated.to_csv('repeated_ANOVA_df.csv')
repeated.dtypes


# # Video Overlap Scatter Plots

# In[15]:


for month in months:
    us = us_top50_byMonth.loc[us_top50_byMonth['trending_YearMonth']==month,:]
    gb = gb_top50_byMonth.loc[gb_top50_byMonth['trending_YearMonth']==month,:]
    ca = ca_top50_byMonth.loc[ca_top50_byMonth['trending_YearMonth']==month,:]
    de = de_top50_byMonth.loc[de_top50_byMonth['trending_YearMonth']==month,:]
    fr = fr_top50_byMonth.loc[fr_top50_byMonth['trending_YearMonth']==month,:]
    us_dict = dict(zip(us['title'],us['video_id_count']))
    gb_dict = dict(zip(gb['title'],gb['video_id_count']))
    ca_dict = dict(zip(ca['title'],ca['video_id_count']))
    de_dict = dict(zip(de['title'],de['video_id_count']))
    fr_dict = dict(zip(fr['title'],fr['video_id_count']))

    plt.figure(figsize=(20,5))
    plt.scatter(np.arange(50),us_dict.values(), s = 20*(np.array(list(us_dict.values()))), color = 'purple')
    plt.title(f'{month}_US')
    plt.ylim(0,31)
    plt.yticks([])
    #plt.ylabel('Number of Trending Days')
    plt.xlim(-2,51)
    plt.xticks(np.arange(50), np.arange(1,51,1))
    plt.savefig(f'Scatter_{month}_1.png')
    
    plt.show()
    us_gb_dict = us_dict.copy()
    for video in gb_dict.keys():
        if video in us_gb_dict.keys():
            us_gb_dict[video] = us_gb_dict[video] + gb_dict[video]
    plt.figure(figsize=(20,5))
    plt.scatter(np.arange(50),us_dict.values(), s = 20*np.array(list(us_gb_dict.values())), color = 'purple')
    plt.title(f'{month}_US_GB')
    #plt.ylabel('Number of Trending Days')
    plt.ylim(0,31)
    plt.yticks([])
    plt.xlim(-2,51)
    plt.xticks(np.arange(50), np.arange(1,51,1))
    plt.savefig(f'Scatter_{month}_2.png')
    plt.show()
    us_gb_ca_dict = us_gb_dict.copy()
    for video in ca_dict.keys():
        if video in us_gb_ca_dict.keys():
            us_gb_ca_dict[video] = us_gb_ca_dict[video] + ca_dict[video]
    plt.figure(figsize=(20,5))
    plt.scatter(np.arange(50),us_dict.values(), s = 20*np.array(list(us_gb_ca_dict.values())), color = 'purple')
    plt.title(f'{month}_US_GB_CA')
    #plt.ylabel('Number of Trending Days')
    plt.ylim(0,31)
    plt.yticks([])
    plt.xlim(-2,51)
    plt.xticks(np.arange(50), np.arange(1,51,1))
    plt.savefig(f'Scatter_{month}_3.png')
    plt.show()
    us_gb_ca_de_dict = us_gb_ca_dict.copy()
    for video in de_dict.keys():
        if video in us_gb_ca_de_dict.keys():
            us_gb_ca_de_dict[video] = us_gb_ca_de_dict[video] + de_dict[video]
    plt.figure(figsize=(20,5))
    plt.scatter(np.arange(50),us_dict.values(), s = 20*np.array(list(us_gb_ca_de_dict.values())), color = 'purple')
    plt.title(f'{month}_US_GB_CA_DE')
    #plt.ylabel('Number of Trending Days')
    plt.ylim(0,31)
    plt.yticks([])
    plt.xlim(-2,51)
    plt.xticks(np.arange(50), np.arange(1,51,1))
    plt.savefig(f'Scatter_{month}_4.png')
    plt.show()
    us_gb_ca_de_fr_dict = us_gb_ca_de_dict.copy()
    for video in fr_dict.keys():
        if video in us_gb_ca_de_fr_dict.keys():
            us_gb_ca_de_fr_dict[video] = us_gb_ca_de_fr_dict[video] + fr_dict[video]
    plt.figure(figsize=(20,5))
    plt.scatter(np.arange(50),us_dict.values(), s = 20*np.array(list(us_gb_ca_de_fr_dict.values())), color = 'purple')
    plt.title(f'{month}_US_GB_CA_DE_FR')
    #plt.ylabel('Number of Trending Days')
    plt.ylim(0,31)
    plt.yticks([])
    plt.xlim(-2,51)
    plt.xticks(np.arange(50), np.arange(1,51,1))
    plt.savefig(f'Scatter_{month}_5.png')
    plt.show()


# In[16]:



us = us_top50_byMonth.loc[us_top50_byMonth['trending_YearMonth']=='2018-2',:]
gb = gb_top50_byMonth.loc[gb_top50_byMonth['trending_YearMonth']=='2018-2',:]
ca = ca_top50_byMonth.loc[ca_top50_byMonth['trending_YearMonth']=='2018-2',:]
de = de_top50_byMonth.loc[de_top50_byMonth['trending_YearMonth']=='2018-2',:]
fr = fr_top50_byMonth.loc[fr_top50_byMonth['trending_YearMonth']=='2018-2',:]
us_dict = dict(zip(us['title'],us['video_id_count']))
gb_dict = dict(zip(gb['title'],gb['video_id_count']))
ca_dict = dict(zip(ca['title'],ca['video_id_count']))
de_dict = dict(zip(de['title'],de['video_id_count']))
fr_dict = dict(zip(fr['title'],fr['video_id_count']))


plt.figure(figsize=(20,5))
plt.scatter(np.arange(50),us_dict.values(), s = 40*(np.array(list(us_dict.values()))), color = 'purple')
plt.title('2018-2_US')
plt.ylim(0,31)
plt.yticks([])
#plt.ylabel('Number of Trending Days')
plt.xlim(-2,51)
plt.xticks(np.arange(50), np.arange(1,51,1))
plt.savefig('Scatter_CountryPair_US_2018-2.png')
plt.show()

us_fr_dict = us_dict.copy()
for video in fr_dict.keys():
    if video in us_fr_dict.keys():
        us_fr_dict[video] = us_fr_dict[video] + fr_dict[video]
plt.figure(figsize=(20,5))
plt.scatter(np.arange(50),us_dict.values(), s = 40*np.array(list(us_fr_dict.values())), color = 'purple')
plt.title('2018-2_US_FR')
#plt.ylabel('Number of Trending Days')
plt.ylim(0,31)
plt.yticks([])
plt.xlim(-2,51)
plt.xticks(np.arange(50), np.arange(1,51,1))
plt.savefig('Scatter_CountryPair_US_FR_2018-2.png')
plt.show()


plt.figure(figsize=(20,5))
plt.scatter(np.arange(50),ca_dict.values(), s = 50*(np.array(list(ca_dict.values()))), color = 'purple')
plt.title('2018-2_CA')
plt.ylim(0,31)
plt.yticks([])
#plt.ylabel('Number of Trending Days')
plt.xlim(-2,51)
plt.xticks(np.arange(50), np.arange(1,51,1))
plt.savefig('Scatter_CountryPair_CA_2018-2.png')
plt.show()

ca_de_dict = ca_dict.copy()
for video in de_dict.keys():
    if video in ca_de_dict.keys():
        ca_de_dict[video] = ca_de_dict[video] + de_dict[video]
plt.figure(figsize=(20,5))
plt.scatter(np.arange(50),ca_dict.values(), s = 50*np.array(list(ca_de_dict.values())), color = 'purple')
plt.title('2018-2_CA_DE')
#plt.ylabel('Number of Trending Days')
plt.ylim(0,31)
plt.yticks([])
plt.xlim(-2,51)
plt.xticks(np.arange(50), np.arange(1,51,1))
plt.savefig('Scatter_CountryPair_CA_DE_2018-2.png')
plt.show()

