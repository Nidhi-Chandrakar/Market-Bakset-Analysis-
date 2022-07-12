#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


missing_value_formats = ["n.a.","?", "??","???","NA","n/a", "na", "--"]
df = pd.read_excel("/Users/himan/OneDrive/Desktop/Data Mining/Course_work/Online Retail.xlsx", na_values = missing_value_formats)
print(df.isnull().sum())


# In[3]:


#Percentage of missing values is
print(df.isnull().sum()*100/len(df))


# In[5]:


#dropping cust_id column
df = df.drop(columns = 'CustomerID')
df['Description'] =df['Description'].str.strip()


# In[6]:


#Dropping Null values
df = df.dropna()


# In[7]:


#Finding and dropping dupliactes
df.duplicated().sum()
df = df.drop_duplicates()
df.shape


# In[8]:


#Deleting all the canclled transactions
data=df[df['Quantity']>=0]
data.info()


# In[9]:


region_prod = data.groupby(['Country'])['InvoiceNo'].count().sort_values(ascending = False).reset_index().rename(columns={'InvoiceNo':'No. of Transactions'})
region_prod.head(5)


# In[10]:


import matplotlib.pyplot as plt
region_prod.plot(kind='bar',x='Country',y='No. of Transactions')


# In[30]:


#generating UK_basket
UK_basket = (data[data['Country']== 'United Kingdom'])
UK_basket.head(5)


# In[31]:


UK_basket['Description'].value_counts()


# In[32]:


UK_basket['Description'].value_counts()[:20].plot(kind='bar')


# In[33]:


# minimum support can be select using expected transaction per period
def determine_min_support(df, expected_txn_per_day):
    min_dt = pd.to_datetime(df['InvoiceDate'].min(), format='%Y%m%d', errors='coerce')
    max_dt = pd.to_datetime(df['InvoiceDate'].max(), format='%Y%m%d', errors='coerce')
    delta = max_dt - min_dt

    actual_txn = len(df)
    expected_txn = expected_txn_per_day * delta.days
    
    return expected_txn / actual_txn

# in this case, select only product with a least 40 transactions
min_sup = determine_min_support(UK_basket,40)
min_sup = round(min_sup,2)
print ('Minimun support of the UK basket is: ', min_sup)


# In[15]:


UK_basket = UK_basket.groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
UK_basket.shape


# In[16]:


UK_basket.shape


# In[17]:


#Encoding
def encode_units(x):
    if x<=0:
        return 0
    if x>=1:
        return 1

UK_basket=UK_basket.applymap(encode_units)
UK_basket


# In[18]:


#Considering only the transactions having more or two item
UK_basket_atleast_2items = UK_basket[(UK_basket>0).sum(axis =1)>=2]
UK_basket_atleast_2items


# In[22]:


#Frequently bought items
UK_frequent_itemsets = apriori(UK_basket_atleast_2items, min_support = 0.03, use_colnames=True)
UK_frequent_itemsets['length'] = UK_frequent_itemsets['itemsets'].apply(lambda x: len(x))
UK_frequent_itemsets.head(10)


# In[25]:


#with minimum support: 0.03 and minimum lift: 1, there are 48 rules
UK_rules = association_rules(UK_frequent_itemsets, metric="lift", min_threshold=1).sort_values('lift', ascending = False).reset_index(drop = True)
print('there are ',len(UK_rules),'rules for the UK basket')
UK_rules


# In[26]:


UK_rules_distinct=UK_rules[UK_rules['antecedent support']>UK_rules['consequent support']]
print('there are ',len(UK_rules_distinct),'distinct rules for the France basket')
UK_rules_distinct


# In[27]:


import warnings
warnings.filterwarnings('ignore')


# In[28]:


def draw_graph(rules, rules_to_show):
  import networkx as nx  
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)    
 #strs=['R0', 'R1', 'R2', 'R3']
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
   
   
  for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
     
    for a in rules.iloc[i]['antecedents']:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 4)
       
    for c in rules.iloc[i]['consequents']:
             
            G1.add_nodes_from([a])
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=4)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')       
 
 
   
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=2)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.15
  nx.draw_networkx_labels(G1, pos)
  plt.show()

#Calling function with 5 rules
draw_graph(UK_rules, 5)


# In[34]:


#generating Germany_basket
Germany_basket = (data[data['Country']== 'Germany'])
Germany_basket.head(5)


# In[39]:


Germany_basket['Description'].value_counts()
Germany_basket['Description'].value_counts()[:20].plot(kind='bar')


# In[36]:


Germany_basket['Description'].value_counts()


# In[41]:


# minimum support can be select using expected transaction per period
def determine_min_support(df, expected_txn_per_day):
    min_dt = pd.to_datetime(df['InvoiceDate'].min(), format='%Y%m%d', errors='coerce')
    max_dt = pd.to_datetime(df['InvoiceDate'].max(), format='%Y%m%d', errors='coerce')
    delta = max_dt - min_dt

    actual_txn = len(df)
    expected_txn = expected_txn_per_day * delta.days
    
    return expected_txn / actual_txn


# In[43]:


# in this case, select only product with a least 40 transactions
min_sup = determine_min_support(Germany_basket,70)
min_sup = round(min_sup,2)
print ('Minimun support of the Germany basket is: ', min_sup)


# In[44]:


Germany_basket = Germany_basket.groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
Germany_basket.shape

Germany_basket.shape


# In[45]:


Germany_basket=Germany_basket.applymap(encode_units)
Germany_basket
Germany_basket.drop('POSTAGE', inplace = True, axis=1)


# In[46]:


#Considering only the transactions having more or two item
Germany_basket_atleast_2items = Germany_basket[(Germany_basket>0).sum(axis =1)>=2]
Germany_basket_atleast_2items


# In[47]:


#Frequently bought items
Germany_frequent_itemsets = apriori(Germany_basket_atleast_2items, min_support = 0.05, use_colnames=True)
Germany_frequent_itemsets['length'] = Germany_frequent_itemsets['itemsets'].apply(lambda x: len(x))
Germany_frequent_itemsets.head(10)


# In[48]:


#with minimum support: 0.05 and minimum lift: 1
Germany_rules = association_rules(Germany_frequent_itemsets, metric="lift", min_threshold=1).sort_values('lift', ascending = False).reset_index(drop = True)
print('there are ',len(Germany_rules),'rules for the Germany basket')
Germany_rules


# In[51]:


Germany_rules_final=Germany_rules[Germany_rules['antecedent support']>Germany_rules['consequent support']]
print('there are ',len(Germany_rules_final),'distinct rules for the France basket')
Germany_rules_final


# In[52]:


def draw_graph(Germany_rules, Germany_rules_to_show):
  import networkx as nx  
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)    
 #strs=['R0', 'R1', 'R2', 'R3']
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
   
   
  for i in range (Germany_rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
     
    for a in Germany_rules.iloc[i]['antecedents']:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 4)
       
    for c in Germany_rules.iloc[i]['consequents']:
             
            G1.add_nodes_from([a])
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=4)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')       
 
 
   
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=2)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.10
  nx.draw_networkx_labels(G1, pos)
  plt.show()

#Calling function with 10 Germany_rules
draw_graph(Germany_rules_final, 5)


# In[54]:


#generating France_basket
France_basket = (data[data['Country']== 'France'])
France_basket


# In[55]:


France_basket['Description'].value_counts()
France_basket['Description'].value_counts()[:20].plot(kind='bar')


# In[56]:


France_basket = France_basket.groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
France_basket


# In[57]:


#Encoding
France_basket=France_basket.applymap(encode_units)
France_basket


# In[58]:


#dropping postage
France_basket.drop('POSTAGE', inplace = True, axis=1)


# In[59]:


#Slecting all the transactis having two or more items
France_basket_min_2items = France_basket[(France_basket>0).sum(axis =1)>=2]
France_basket_min_2items
France_basket_min_2items.shape


# In[60]:


France_frequent_itemsets = apriori(France_basket_min_2items, min_support = 0.05, use_colnames=True)
France_frequent_itemsets['length'] = France_frequent_itemsets['itemsets'].apply(lambda x: len(x))
France_frequent_itemsets


# In[61]:


France_rules = association_rules(France_frequent_itemsets, metric="lift", min_threshold=1).sort_values('lift', ascending = False).reset_index(drop = True)
print('there are ',len(France_rules),'rules for the France basket')
France_rules


# In[ ]:


def draw_graph(Germany_rules, Germany_rules_to_show):
  import networkx as nx  
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)    
 #strs=['R0', 'R1', 'R2', 'R3']
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
   
   
  for i in range (Germany_rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
     
    for a in Germany_rules.iloc[i]['antecedents']:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 4)
       
    for c in Germany_rules.iloc[i]['consequents']:
             
            G1.add_nodes_from([a])
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=4)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')       
 
 
   
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=2)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.10
  nx.draw_networkx_labels(G1, pos)
  plt.show()

#Calling function with 10 Germany_rules
draw_graph(Germany_rules_final, 5)

