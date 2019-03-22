import pandas as pd
import numpy as np
from io import BytesIO
import requests
from bs4 import BeautifulSoup

head={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}

url = 'https://www.bls.gov/cps/cpsaat01.xlsx'
download=requests.get(url, headers=head)
df =  pd.read_excel(BytesIO(download.content))
df.columns = ['Date', 'Civilian noninsti-tutional population', 'Total', 'Percent of population', 'Employed-Total', 'Employed-Percent of Population', 'Employed-Agriculture', "Employed-Nonagricultural industries", 'Unemployed-Number', 'Unemployed_Percent_of_labor_force', 'Not in labor force']
df = df.drop([0, 1, 2, 3, 4, 5, 6, df.shape[0]-1, df.shape[0]-2])
df.set_index('Date', inplace=True)
d = df.Unemployed_Percent_of_labor_force
d = d.to_frame()
d.columns = ['Value']

d.head()
d.reset_index(inplace=True)
print(d)

