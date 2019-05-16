import math
import numpy
import numpy.random as nrand

"""
Note - for some of the metrics the absolute value is returns. This is because if the risk (loss) is higher we want to
discount the expected excess return from the portfolio by a higher amount. Therefore risk should be positive.
"""


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

def ReadData():
    ETF_File=pd.read_csv('Commodity ETF List (125).csv')

    ChooseETFSymbol=list()
    for i in range(len(ETF_File['Inception'])):
        if int(ETF_File['Inception'][i][:4])<2015:
           ChooseETFSymbol.append(ETF_File['Symbol'][i]) 
    ChooseETFSymbol.append("SPY")
    print(ChooseETFSymbol)


    startTime=[[2015,12,31],[2016,4,1],[2016,7,1],[2016,10,1],[2017,1,1],[2017,4,1],[2017,7,1],[2017,10,1],[2018,1,1],[2018,4,1],[2018,7,1],[2018,10,1],[2019,1,1]]
    endTime=[[2016,3,30],[2016,6,30],[2016,9,30],[2016,12,30],[2017,3,30],[2017,6,30],[2017,9,30],[2017,12,30],[2018,3,30],[2018,6,30],[2018,9,30],[2018,12,30],[2019,3,30]]

    AllETFDataFrame=pd.DataFrame()
    ETFcount=0
    for ETF in ChooseETFSymbol[-3:]:

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
    return AllETFDataFrame


def vol(returns):
    # Return the standard deviation of returns
    return numpy.std(returns)


def beta(returns, market):
    # Create a matrix of [returns, market]
    m = numpy.matrix([returns, market])
    # Return the covariance of m divided by the standard deviation of the market returns
    return numpy.cov(m)[0][1] / numpy.std(market)


def lpm(returns, threshold, order):
    
    threshold_array = numpy.empty(len(returns))
    threshold_array.fill(threshold)
    
    diff = threshold_array - returns
    
    diff = diff.clip(min=0)
    
    return numpy.sum(diff ** order) / len(returns)


def hpm(returns, threshold, order):
    
    threshold_array = numpy.empty(len(returns))
    threshold_array.fill(threshold)
    
    diff = returns - threshold_array
    
    diff = diff.clip(min=0)
    
    return numpy.sum(diff ** order) / len(returns)


def var(returns, alpha):
    
    sorted_returns = numpy.sort(returns)
    
    index = int(alpha * len(sorted_returns))
    
    return abs(sorted_returns[index])


def cvar(returns, alpha):
    
    sorted_returns = numpy.sort(returns)
    
    index = int(alpha * len(sorted_returns))
    
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    
    return abs(sum_var / index)


def prices(returns, base):
    
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return numpy.array(s)


def dd(returns, tau):
    # Returns the draw-down given time period tau
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    # Find the maximum drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    # Drawdown should be positive
    return abs(drawdown)


def max_dd(returns):
   
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # Max draw-down should be positive
    return abs(max_drawdown)


def average_dd(returns, periods):
    
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def average_dd_squared(returns, periods):
    
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = math.pow(dd(returns, i), 2.0)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def treynor_ratio(er, returns, market, rf):
    return (er - rf) / beta(returns, market)


def sharpe_ratio(er, returns, rf):
    return (er - rf) / vol(returns)


def information_ratio(returns, benchmark):
    diff = returns - benchmark
    return numpy.mean(diff) / vol(diff)


def modigliani_ratio(er, returns, benchmark, rf):
    np_rf = numpy.empty(len(returns))
    np_rf.fill(rf)
    rdiff = returns - np_rf
    bdiff = benchmark - np_rf
    return (er - rf) * (vol(rdiff) / vol(bdiff)) + rf


def excess_var(er, returns, rf, alpha):
    return (er - rf) / var(returns, alpha)


def conditional_sharpe_ratio(er, returns, rf, alpha):
    return (er - rf) / cvar(returns, alpha)


def omega_ratio(er, returns, rf, target=0):
    return (er - rf) / lpm(returns, target, 1)


def sortino_ratio(er, returns, rf, target=0):
    return (er - rf) / math.sqrt(lpm(returns, target, 2))


def kappa_three_ratio(er, returns, rf, target=0):
    return (er - rf) / math.pow(lpm(returns, target, 3), float(1/3))


def gain_loss_ratio(returns, target=0):
    return hpm(returns, target, 1) / lpm(returns, target, 1)


def upside_potential_ratio(returns, target=0):
    return hpm(returns, target, 1) / math.sqrt(lpm(returns, target, 2))


def calmar_ratio(er, returns, rf):
    return (er - rf) / max_dd(returns)


def sterling_ration(er, returns, rf, periods):
    return (er - rf) / average_dd(returns, periods)


def burke_ratio(er, returns, rf, periods):
    return (er - rf) / math.sqrt(average_dd_squared(returns, periods))


def test_risk_metrics():
    # This is just a testing method
    r = nrand.uniform(-1, 1, 50)
    m = nrand.uniform(-1, 1, 50)
    print("vol =", vol(r))
    print("beta =", beta(r, m))
    print("hpm(0.0)_1 =", hpm(r, 0.0, 1))
    print("lpm(0.0)_1 =", lpm(r, 0.0, 1))
    print("VaR(0.05) =", var(r, 0.05))
    print("CVaR(0.05) =", cvar(r, 0.05))
    print("Drawdown(5) =", dd(r, 5))
    print("Max Drawdown =", max_dd(r))


def test_risk_adjusted_metrics():
    ReadData()

    #模擬市場m與投組r的報酬
    r = nrand.uniform(-1, 1, 50)
    m = nrand.uniform(-1, 1, 50)
    # Expected return
    #投組期望報酬
    e = numpy.mean(r)
    # 無風險利率
    f = 0.06
    
    print("Treynor Ratio =", treynor_ratio(e, r, m, f))
    print("Sharpe Ratio =", sharpe_ratio(e, r, f))
    print("Information Ratio =", information_ratio(r, m))
    # Risk-adjusted return based on Value at Risk
    print("Excess VaR =", excess_var(e, r, f, 0.05))
    print("Conditional Sharpe Ratio =", conditional_sharpe_ratio(e, r, f, 0.05))
    print("")
    # Risk-adjusted return based on Lower Partial Moments
    print("Omega Ratio =", omega_ratio(e, r, f))
    print("Sortino Ratio =", sortino_ratio(e, r, f))
    print("Kappa 3 Ratio =", kappa_three_ratio(e, r, f))
    print("Gain Loss Ratio =", gain_loss_ratio(r))
    print("Upside Potential Ratio =", upside_potential_ratio(r))
    # Risk-adjusted return based on Drawdown risk
    print("Calmar Ratio =", calmar_ratio(e, r, f))
    print("Sterling Ratio =", sterling_ration(e, r, f, 5))
    print("Burke Ratio =", burke_ratio(e, r, f, 5))

def ComputeOmega():
    AllData=ReadData()

    #模擬市場m與投組r的報酬
    #r = nrand.uniform(-1, 1, 50)
    #m = nrand.uniform(-1, 1, 50)
    print(AllData["LD"].values[1:])
    r1=(AllData["LD"].values[1:]-AllData["LD"].values[:-1])/AllData["LD"].values[:-1]
    r2=(AllData["DDP"].values[1:]-AllData["DDP"].values[:-1])/AllData["DDP"].values[:-1]
    m=(AllData["SPY"].values[1:]-AllData["SPY"].values[:-1])/AllData["SPY"].values[:-1]
    #m=AllData["SPY"].values.pct_change()
    # Expected return
    #投組期望報酬
    e = numpy.mean(r1)
    # 無風險利率
    f = 0.06
    
    
    # Risk-adjusted return based on Lower Partial Moments
    print("Omega Ratio =", omega_ratio(e, r1, f))
from functools import reduce
def str2float(s):
    def fn(x,y):
        return x*10+y
    n=s.index('.')
    s1=list(map(int,[x for x in s[:n]]))
    s2=list(map(int,[x for x in s[n+1:]]))
    return reduce(fn,s1) + reduce(fn,s2)/10**len(s2)
print('\'123.4567\'=',str2float('123.4567')) 
def MyDataFrameProcess(Strlist):
    import numpy as np
    AnsList=list()
    for i in range(len(Strlist)):

        AnsList.append(str2float(Strlist[i]))
    return np.array(AnsList)

def GeneralizedSharpe():

    import numpy as np
    import math
    from scipy.stats import skew
    from scipy.stats import kurtosis
    AllData=ReadData()

    LD=MyDataFrameProcess(AllData["SPY"].values)
    #print(LD)
    #ej3ru84=array(pd.DataFrame(LD).pct_change().values).reshape(len(LD))
    Asset_Return=(LD[1:]-LD[:-1])/LD[:-1]
    print(Asset_Return)
    mean = np.mean(Asset_Return)
    standard = np.std(Asset_Return)
    variance = np.var(Asset_Return)
    S = skew(Asset_Return)
    K = kurtosis(Asset_Return)
    print(S,K,K>3+(5/3)*S*S)
    a = 3*math.sqrt(3*K-4*(S**2)-9)/variance*(3*K-5*(S**2)-9)
    b = 3*S/standard*(3*K-5*(S**2)-9)
    n = mean-(3*S*standard)/(3*K-4*(S**2)-9)
    d = 3*standard*math.sqrt(3*K-4*(S**2)-9)/(3*K-5*(S**2)-9)
    phi = math.sqrt((a**2)-(b**2))

    rf = 0.02 # 定存利率
    lamda = 0.5
    astar = 1/lamda*(b+(a*(n-rf)/math.sqrt((d**2)+((n-rf)**2))))
    asksr = math.sqrt(2*(lamda*astar*(n-rf)-d*(phi-math.sqrt((a**2)-((b-lamda*astar)**2)))))

    print("asksr",asksr,a,b,n,d,phi)
    return asksr
def ComputeRiskiness():
    from scipy.optimize import fsolve,leastsq,root
    import numpy as np
    AllData=ReadData()


    def f(alpha):
        import numpy as np
        sum=0
        for i in range(len(gamble)):
            sum=np.exp(-gamble[i]*alpha)+sum

        print("sum",sum,len(gamble))
        return sum-len(gamble)
    Asset=MyDataFrameProcess(AllData["SPY"].values)
    print(Asset)
    #ej3ru84=array(pd.DataFrame(LD).pct_change().values).reshape(len(LD))
    #Asset_Return=(Asset[1:]-Asset[:-1])/Asset[:-1]
    Asset_Return=((Asset[1:]-Asset[:-1])/Asset[:-1])/10000
    
    print(Asset_Return)


    gamble=Asset_Return
    
    #print(f(alpha0))

    result=fsolve(f,0.0003)
    print("Result",result)
    print("Test",f(result))
    print("Riskiness",np.exp(-result))




if __name__ == "__main__":
    #ComputeOmega()
    ComputeRiskiness()
    #nj04asksr()
    # test_risk_metrics()
    # test_risk_adjusted_metrics()