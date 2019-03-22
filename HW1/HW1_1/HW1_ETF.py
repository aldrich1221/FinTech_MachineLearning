
import pandas as pd
from pyquery import PyQuery as pq
import datetime
import time
import enum
def Month2int(monthstr):
	if monthstr=='Jan':
		return 1
	elif monthstr=='Feb':
		return 2
	elif monthstr=='Mar':
		return 3
	elif monthstr=='Apr':
		return 4
	elif monthstr=='May':
		return 5
	elif monthstr=='Jun':
		return 6
	elif monthstr=='Jul':
		return 7
	elif monthstr=='Aug':
		return 8
	elif monthstr=='Sep':
		return 9
	elif monthstr=='Oct':
		return 10
	elif monthstr=='Nov':
		return 11
	elif monthstr=='Dec':
		return 12
	else:
		return 0

  
ETF_File=pd.read_csv('Commodity ETF List (125).csv')

ChooseETFSymbol=list()
for i in range(len(ETF_File['Inception'])):
    if int(ETF_File['Inception'][i][:4])<2015:
       ChooseETFSymbol.append(ETF_File['Symbol'][i]) 
        
print(ChooseETFSymbol)


startTime=[[2015,12,31],[2016,4,1],[2016,7,1],[2016,10,1],[2017,1,1],[2017,4,1],[2017,7,1],[2017,10,1],[2018,1,1],[2018,4,1],[2018,7,1],[2018,10,1],[2019,1,1]]
endTime=[[2016,3,30],[2016,6,30],[2016,9,30],[2016,12,30],[2017,3,30],[2017,6,30],[2017,9,30],[2017,12,30],[2018,3,30],[2018,6,30],[2018,9,30],[2018,12,30],[2019,3,30]]

AllETFDataFrame=pd.DataFrame()
ETFcount=0
for ETF in ChooseETFSymbol:

	DateList=list()
	AdjCloseList=list()
	for TimePeriod in range(len(startTime)):

		startdatatime = datetime.datetime(startTime[TimePeriod][0], startTime[TimePeriod][1], startTime[TimePeriod][2], 0, 0, 0)
		starttimestamp = time.mktime(startdatatime.timetuple())
		
		enddatatime = datetime.datetime(endTime[TimePeriod][0], endTime[TimePeriod][1], endTime[TimePeriod][2], 0, 0, 0)
		endtimestamp = time.mktime(enddatatime.timetuple())
		

		
		try:
		#Website=pq('https://finance.yahoo.com/quote/'+ETF+'/history?period1=1451491200&period2=1553097600&interval=1d&filter=history&frequency=1d', headers={'user-agent': 'pyquery'})
			Website=pq('https://finance.yahoo.com/quote/'+ETF+'/history?period1='+str(int(starttimestamp))+'&period2='+str(int(endtimestamp))+'&interval=1d&filter=history&frequency=1d', headers={'user-agent': 'pyquery'})
		#Website=pq('https://finance.yahoo.com/quote/GCC/history?period1=1451491200&period2=1553097600&interval=1d&filter=history&frequency=1d', headers={'user-agent': 'pyquery'})
		
		except:
			print("Cannot send requests")
			
		
		Tbody=Website('tbody')
		Trs = Tbody('tr')
		
		for Tr in Trs.items():
			
			
			datas=Tr('span')
			datacount=0
			
			for data in datas.items():
				

				if datacount==4:
					pricedata=data.text()
					
					
				if datacount==0:
					DateStrings=data.text().replace(",","").split(" ")
					
						
					dataDatetime=datetime.date(int(DateStrings[2]),Month2int(DateStrings[0]),int(DateStrings[1]))
				datacount=datacount+1
			
			AdjCloseList.append(pricedata)
			DateList.append(dataDatetime)

	ETFDataFrame=pd.DataFrame(AdjCloseList,columns=[str(ETF)],index=DateList).sort_index()
	#print(ETFDataFrame)
	if ETFcount==0:
		AllETFDataFrame=ETFDataFrame

	else:
		
		AllETFDataFrame=AllETFDataFrame.join(ETFDataFrame,how='left',sort=True)
		
	ETFcount=ETFcount+1
print(AllETFDataFrame)
 	


