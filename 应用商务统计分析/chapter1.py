'''

《应用商务统计分析》 
第一章 线性回归（上市公司净资产收益率预测分析）

1. 数据采集
确认采样信息，包括自变量和因变量、时间范围、数据来源（优矿数据平台）。

2. 数据清洗
了解数据，对异常数据进行适当的清洗处理。

3. 描述性分析
通过多角度，进一步熟悉数据，对数据有一定的认识。

4. 全模型分析
在描述性分析的基础上，尝试通过最小二乘法建立多元线性回归模型，在得到估计值之后，需进行必要的检验与评价（拟合程度的测定、估计标准误差、回归方程的显著性检验、回归系数的显著性检验、多重共线性判别），以决定模型是否可以应用。

5. 模型选择
为了提高模型预测能力，需对自变量进行选择，选择变量的方法主要有AIC和BIC。

6. 模型预测
评估模型在新数据集上的预测效果。

'''



import pandas as pd
import numpy as np
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error


# print配置
str1 = '\n'*4 + '#'*80 
str2 = '\n'*2 + '-'*80 


def get_data(beginDate, endDate):
    '''
    一、数据采集（优矿数据平台）
    输入：开始日，结束日
    输出：原始数据集
    '''

    # 1. 股票基本信息
    df = DataAPI.EquGet(secID=u"",ticker=u"",equTypeCD=u"A",listStatusCD=u"L",field=['ticker', 'secShortName', 'listDate'],pandas="1")
    df = df[df.listDate<=beginDate]
    tickerlst = df.ticker.tolist()

    # 2. 财务数据
    # 2.1 合并资产负债表(所有会计期末最新披露)
    fdmt1 = DataAPI.FdmtBSAllLatestGet(ticker=tickerlst,secID=u""
                                    ,reportType=u"A",endDate=u"",beginDate=beginDate,year=""
                                    ,field=['ticker', 'endDate', 'inventories', 'TAssets', 'TLiab']
                                    ,pandas="1")
    # 2.2 业绩快报(所有会计期末最新披露)
    fdmt2 = DataAPI.FdmtEeAllLatestGet(ticker=tickerlst,secID=u""
                                       ,reportType=u"A",endDate=u"",beginDate=beginDate
                                       ,field=['ticker', 'endDate', 'ROE']
                                       ,pandas="1")
    # 2.3 财务指标—营运能力(根据所有会计期末最新披露数据计算)
    fdmt3 = DataAPI.FdmtIndiTrnovrGet(ticker=tickerlst,secID=""
                                     ,endDate="",beginDate=beginDate,beginYear="",endYear=u"",reportType=u"A"
                                     ,field=['ticker', 'endDate', 'taTurnover']
                                     ,pandas="1")
    # 2.4 财务指标—成长能力(根据所有会计期末最新披露数据计算)
    fdmt4 = DataAPI.FdmtIndiGrowthGet(ticker=tickerlst,secID=""
                                      ,endDate="",beginDate=beginDate,beginYear=u"",endYear=u"",reportType=u"A"
                                      ,field=['ticker', 'endDate', 'revenueYOY']
                                      ,pandas="1")
    # 2.5 财务指标-利润表结构分析(根据所有会计期末最新披露数据计算)
    fdmt5 = DataAPI.FdmtIndiStctISGet(ticker=tickerlst,secID=""
                                      ,endDate="",beginDate=beginDate,beginYear=u"",endYear=u"",reportType=u"A"
                                      ,field=['ticker', 'endDate', 'opTR']
                                      ,pandas="1")
    # 2.6 财务指标-收现能力(根据所有会计期末最新披露数据计算)
    fdmt6 = DataAPI.FdmtIndiCashGet(ticker=tickerlst,secID=""
                                    ,endDate="",beginDate=beginDate,beginYear=u"",endYear=u"",reportType=u"A"
                                    ,field=['ticker', 'endDate', 'arR']
                                    ,pandas="1")
    # 2.7 财务数据合并
    fdmt = pd.merge(fdmt1, fdmt2)
    fdmt = pd.merge(fdmt, fdmt3)
    fdmt = pd.merge(fdmt, fdmt4)
    fdmt = pd.merge(fdmt, fdmt5)
    fdmt = pd.merge(fdmt, fdmt6)
    fdmt['ROE'] = fdmt.ROE / 100
    fdmt['arR'] = fdmt.arR / 100
    fdmt['opTR'] = fdmt.opTR / 100
    fdmt['revenueYOY'] = fdmt.revenueYOY / 100
    fdmt['lev'] = fdmt.TLiab/fdmt.TAssets
    fdmt['inv'] = fdmt.inventories/fdmt.TAssets
    fdmt['logTAssets'] = np.log(fdmt.TAssets)
    fdmt['year'] =  fdmt.endDate.str.slice(0,4)
    fdmt['ROEf'] = fdmt.sort_values(by=['ticker', 'endDate']).groupby('ticker').ROE.shift(-1)
    fdmt = fdmt[fdmt.ROEf.notnull()]

    # 3. 行情数据
    pbdata = pd.DataFrame()
    m = 100
    st = 0
    for sub in range(0, len(tickerlst), m):
        sublst = tickerlst[sub:sub+m]
        mkt = DataAPI.MktEqudGet(secID=u"",ticker=sublst
                                ,tradeDate=u"",beginDate=beginDate,endDate=endDate,isOpen="1"
                                ,field=['ticker', 'tradeDate', 'PB']
                                ,pandas="1")
        mkt['year'] = mkt.tradeDate.str.slice(0,4)
        mkt = mkt.groupby(['year', 'ticker'], as_index=False).PB.mean()
        pbdata = pd.concat((pbdata, mkt), axis=0)

    # 4. 数据合并
    alldata = pd.merge(fdmt[['ticker', 'year', 'ROE', 'taTurnover', 'lev', 'arR', 'opTR', 'revenueYOY', 'inv', 'logTAssets', 'ROEf']], pbdata)
    print(alldata.year.value_counts().sort_index())
  
    return alldata


def clean_data(df, thresholddict, splityeardict, validcolsdict):
    '''
    二、数据清洗
    通过观察步骤一中原始数据：
    1. 数据类型
    2. 异常情况
    3. 缺失等情况
    制定相应的清洗规则如下：
    1. 缺失值剔除：若特征或样本的缺失率大于等于缺失值阈值，则剔除该特征或者样本；
    2. 缺失值填补：使用训练集中位数填补；
    3. 极值处理：使用分位数替换的方法进行极值处理，即将大于百分位上限阈值和小于下限阈值的值，分别用相应的上下限百分位阈值替换；
    参数说明：
    输入
    1. df：原始DataFrame数据集；
    2. thresholddict：阈值字典，包含nanlmt（缺失率）、lowlmt（百分位下限）、uplmt（百分位上限）；
    3. splityeardict：数据集划分年份字典， 包含trainyear、testyear；
    4. validcolsdict：有效特征集；
    输出
    1. modeldata：清洗后的DataFrame数据集；
    '''
    
    # 参数
    modeldata = df.copy()
    nanlmt = thresholddict['nanlmt']
    trainyear = splityeardict['trainyear']
    lowlmt = thresholddict['lowlmt']
    uplmt = thresholddict['uplmt'] 
    validcolslst = list(validcolsdict.keys())
    print(str2 + 'Begin')
    print(np.round(modeldata.describe().T, 3))
    
    # 缺失值剔除(行与列)
    rescols = modeldata.columns[(modeldata.isnull().sum(axis=0) / modeldata.shape[0]) < nanlmt]
    resindexs = modeldata.index[modeldata.isnull().sum(axis=1) / modeldata.shape[1] < nanlmt]
    modeldata = modeldata.loc[resindexs, rescols]
    
    # 数据切分
    mask = modeldata.year.isin(trainyear)
    
    # 缺失值填补
    med = modeldata[mask].median()
    modeldata = modeldata.fillna(med)
    
    # 极值处理
    lowlmts =modeldata.loc[mask, validcolslst].quantile(lowlmt)
    uplmts =modeldata.loc[mask, validcolslst].quantile(uplmt)
    for i in validcolslst:
        modeldata[i] = modeldata[i].clip(lowlmts[i], uplmts[i])
        
    print(str2 + 'End')
    print(np.round(modeldata.describe().T, 3))
    
    return modeldata


def corr_plot(corrdata):  
    '''
    相关性热力图
    输入：corr矩阵
    '''   
    cmap = 'seismic'
    fig = plt.figure(figsize = (10, 8), dpi = 300)
    plt.tick_params(labelsize=10)
    # 使用mask不显示相关系数矩阵右上三角
    mask = np.zeros_like(corrdata)
    mask[np.triu_indices_from(mask)] = True
    ax = sns.heatmap(corrdata, cmap=cmap, mask=mask, linewidths=2, annot=True, fmt='.2f', annot_kws={'size':12})
    ax.set_title('Corr', fontweight ='bold', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()
    plt.close()
    
   
def desc_data(df, target, validcolsdict):
    '''
    三、描述性分析
    主要从以下几个方面，进行数据的描述性分析，让我们对数据有一个初步的认识：
    1. 基本描述：数量、均值、标准差、最小值、百分位25%、百分位50%、百分位75%、最大值；
    2. 单变量分布情况及稳定性
    3. 两个变量间的相关性
    4. 自变量与因变量的关系
    输入：
    1. df：DataFrame数据集；
    2. target：因变量；
    3. validcolsdict：有效自变量字典；
    '''
    
    # 处理
    validcolslst = list(validcolsdict.keys()) # 有效变量
    sns.set_style("white")
    mpl.rcParams['axes.color_cycle'] = ['r', 'g', 'b', 'c']# 系列颜色
    
    # 1. 数据概况
    print(str2 + 'Desc')
    descidx = ['count', 'mean', 'std', 'min', '50%', 'max']
    desc = np.round(df[validcolslst].describe().T[descidx], 3)
    print(desc)
    
    # 2. 单变量分布情况
    print(str2 + 'Kde')
    fig = plt.figure(figsize = (15, 12), dpi = 300)
    col = 3
    row = len(validcolslst) // col
    for i in range(1, row+1):
        for j in range(1, col+1):
            idx = (i - 1)*col + j - 1
            tmpax = fig.add_subplot(row, col, idx+1)
            colname = validcolslst[idx]
            df.groupby('year')[colname].plot.kde(legend=True, ax = tmpax)
            tmpax.set_title(colname, fontweight ='bold', fontsize=16)
    plt.tight_layout()
    plt.show() 
    plt.close()
        
    # 3. 变量间的相关性
    print(str2 + 'Corr')
    corrdata = df[validcolslst].corr()
    corr_plot(corrdata)
    
    # 4. 自变量与因变量的关系
    print(str2 + 'x ＆ ｙ')
    xcols = [i for i in validcolslst if i != target]
    fig = plt.figure(figsize = (15, 12), dpi = 300)
    col = 3
    row = len(xcols) // col
    for i in range(1, row+1):
        for j in range(1, col+1):
            idx = (i - 1)*col + j - 1
            tmpax = fig.add_subplot(row, col, idx+1)
            colname = xcols[idx]
            subcorr = stats.pearsonr(df[target], df[colname])
            df.plot.scatter(x=colname, y=target, ax = tmpax)
            tmpax.set_title('%s & %s : R=%.2f, p=%.2f' % (colname, target, subcorr[0], subcorr[1]), fontweight ='bold', fontsize=16)
    plt.tight_layout()
    plt.show() 
    plt.close()

       
def plot_lr_diagnosis(lm, df, target):
    '''
	线性回归基本前提假设：
	1. 线性：因变量和每个自变量都是线性关系；
	2. 独立性：对于所有的观测值，它们的误差项相互之间是独立的；
	3. 正态性：误差项服从正态分布；
	4. 等方差：所有的误差项具有同样方差。
	
	
    线性回归模型图形诊断：
	1. Residuals vs Fitted（残差拟合图）：
		1） 残差图分析法是一种直观、方便的分析方法。它以残差ei为纵坐标，以其他适宜的变量（如样本拟合值）为横坐标画散点图,主要用来检验是否存在异方差。
		2） 一般情况下，当回归模型满足所有假定时，残差图上的n个点的散布应该是随机的，无任何规律。如果残差图上的点的散布呈现出一定趋势（随横坐标的增大而增大或减小），则可以判断回归模型存在异方差。
		3） 异方差：某一因素或某些因素随着解释变量观测值的变化而对被解释变量产生不同的影响，导致随机误差产生不同方差。当存在异方差时，普通最小二乘估计存在以下问题：
			* 参数估计值虽然是无偏的，但不是最小方差线性无偏估计；
			* 参数的显著性检验失效；
			* 回归方程的应用效果极不理想。
					   
	2. Normal Q-Q（Q-Q图）
		1） Q-Q图主要用来检验样本是否近似服从正态分布。
		2） 对于标准状态分布而言，Q-Q图上的点近似在Y=X直线附近。
		  
	3. Scale-Location（标准化的残差对拟合值）
		此图类似于残差图，只是其纵坐标变为了标准化残差的绝对值开方。
	
	4. Residual vs Leverage（Cook距离图）
		1） Cook距离用来判断强影响点是否为Y的异常值点。
		2） 一般认为当D<0.5时认为不是异常值点；当D>0.5时认为是异常值点。
	
    '''
	
    results = pd.DataFrame({'index': df[target], # y实际值
                            'resids': lm.resid, # 残差
                            'std_resids': lm.resid_pearson, # 方差标准化的残差
                            'fitted': lm.predict() # y预测值
                           })
    fig = plt.figure(figsize = (10, 10), dpi = 100)
    
    # Residual vs Fitted
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(results['fitted'], results['resids'],  'o')
    l = plt.axhline(y = 0, color = 'grey', linestyle = 'dashed')
    ax1.set_xlabel('Fitted values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')

    # Normal Q-Q
    ax2 = fig.add_subplot(2, 2, 2)
    sm.qqplot(results['std_resids'], line='s', ax = ax2)
    ax2.set_title('Normal Q-Q')

    # Scale-Location
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(results['fitted'], abs(results['std_resids'])**.5,  'o')
    ax3.set_xlabel('Fitted values')
    ax3.set_ylabel('Sqrt(|standardized residuals|)')
    ax3.set_title('Scale-Location')

    # Residual vs Leverage
    ax4 = fig.add_subplot(2, 2, 4)
    sm.graphics.influence_plot(lm, criterion = 'Cooks', size = 2, ax = ax4)

    plt.tight_layout()
    

def calc_vif(df):
	'''
	多重共线性（膨胀因子VIF）
	1. 若自变量间存在共线性的现象，可能会降低模型的效率，使得模型难以区分自变量与其他自变量的作用，因此相应的回归系数估计会很不精确。
	2. 通常使用膨胀因子VIF来度量自变量的多重共线性程度，它反映了在多大程度上该自变量所包含的信息已被其他自变量所覆盖。（VIF需小于10）
	'''	
	X = df
	X['as'] = 1 # 添加常数项https://xbuba.com/questions/42658379
	vif = pd.DataFrame()
	vif["features"] = xcols
	vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(len(xcols))]
	return vif
	
	
def forward_selected(data, target, method):
    '''
    向前逐步回归：
	1. 遍历属性的一列子集，选择使模型效果最好的自变量。
	2. 接着寻找与其组合效果最好的第二个自变量，而不是遍历所有的两列子集。
	3. 以此类推，每次遍历时，子集都包含上一次遍历得到的最优子集。这样，每次遍历都会选择一个新的自变量添加到特征集合中，直至特征集合中特征个数不能再增加。
    '''
    variate = set(data.columns) 
    variate.remove(target)  # 原始自变量集
    selected = [] # 最终自变量集
    current_score, best_new_score = float('inf'), float('inf')  # 设置分数的初始值为无穷大（因为aic/bic越小越好）
    # 循环筛选变量
    while variate:
        score_with_variate = [] # 记录遍历过程的分数
		# 遍历自变量
        for candidate in variate:  
            formula = "{}~{}".format(target,"+".join(selected+[candidate]))  # 组合
            if method == 'AIC':
                score = smf.ols(formula=formula, data=data).fit().aic 
            else:
                score=smf.ols(formula=formula, data=data).fit().bic 
            score_with_variate.append((score, candidate))  
        score_with_variate.sort(reverse=True)  # 降序后，取出当前最好模型分数
        best_new_score, best_candidate = score_with_variate.pop()  
        if current_score > best_new_score:  # 如果当前最好模型分数 优于 上一次迭代的最好模型分数，则将对于的自变量加入到selected中
            variate.remove(best_candidate)
            selected.append(best_candidate)  
            current_score = best_new_score  
            # print("score is {},continuing!".format(current_score))  
        else:
            # print("for selection over!")
            break
    formula = "{}~{}".format(target,"+".join(selected))  
    # print("final formula is {}".format(formula))
    model = smf.ols(formula=formula, data=data).fit()
    
    return(model)


def model_report(model):
    '''
    模型报告
	1. 回归方程显著性（F检验）：检验模型的显著性，假设所有的自变量中至少有一个自变量对因变量有重要的解释性作用（H1），当P值小于给定的显著水平α时，则表示拒绝H0；
	2. 回归系数显著性（t检验）：在确定模型显著性后，需确定是哪一个或哪一些自变量有预测能力，则需对自变量逐一进行t检验；
	3. 拟合优度：对模型的拟合优度进行量化的判断，从而或者自变量对因变量的解释力度，一般认为R^2越大，解释效果越好，但是当变量增多时，R^2只会不断增加而不会减少，这时候引入调整后的R^2（变量数的惩罚）；
	4. 残差项标准差
    '''
	print('AIC: %.2F  BIC: %.2f' % (model.aic, model.bic))
    print(model.summary().tables[1])
    print('残差项标准差：%.3f  模型F检验P值：%.3f \n判决系数（R^2）:%.3f  调整的判决系数（R^2）:%.3f'
          % (np.std(model.resid), model.f_pvalue, model.rsquared, model.rsquared_adj))
    
    

	

################ 0. 设置参数 #################
print(str1 + ' 0. 设置参数')

# 有效变量字典
validcolsdict = {
    'ROE': '当年净资产收益率ROE'
    , 'ROEf': '下一年净资产收益率ROE'
    , 'taTurnover':'资产周转率ATO'
    , 'lev':'债务资本比率LEV'
    , 'PB':'市倍率PB'
    , 'arR':'应收账款/营业收入(%)'
    , 'opTR':'营业利润/营业总收入(%)'
    , 'revenueYOY':'营业收入同比(%)'
    , 'inv':'存货/资产总计INV'
    , 'logTAssets':'资产总计对数ASSET'
    }
	
# 因变量
target = 'ROEf' 

# 其他参数
args = {'data_rng':{'begin': '2015-01-01', 'end': '2018-01-01'}
        , 'clean_data_args':{'nanlmt': 0.2, 'lowlmt': 0.025, 'uplmt': 0.975} # 阈值：nanlmt缺失率，lowlmt异常值缩尾下百分位，uplmt异常值缩尾上百分位
        , 'split_data_args':{'trainyear': ['2015'], 'testyear':['2016']} # 数据集划分：trainyear训练集年份，testyear测试集年份
        , 'a':0.05 # t检验显著性水平
       }



################ 1. 数据采集 #################
print(str1 + ' 1. 数据采集')
df = get_data(args['data_rng']['begin']
              , args['data_rng']['end'])



################ 2. 数据清洗 #################
print(str1 + ' 2. 数据清洗')
modeldata = clean_data(df[df.year.isin(np.array(args['split_data_args'].values()).flatten())]
                       , args['clean_data_args']
                       , args['split_data_args']
                       , validcolsdict)



################# 4. 描述性分析 #################
print(str1 + ' 3. 描述性分析')
desc_data(modeldata
          , target
          , validcolsdict)



################# 4. 全模型分析 ################
print(str1 + ' 4. 全模型分析')

# 划分数据集
traindata = modeldata[modeldata.year.isin(args['split_data_args']['trainyear'])]
xcols = [i for i in validcolsdict.keys() if i != target]

# 建立回归模型
formula = '%(y)s ~ %(x)s' % {'y':target, 'x':'+'.join(xcols)} # 多元线性模型方程
lm = smf.ols(formula=formula, data=traindata).fit() # 使用最小二乘进行参数估计

# 模型检验报告
print(str2 + '4.1. 全模型')
selected = list(lm.pvalues.index[lm.pvalues<args['a']]) # t检验显著的特征
model_report(lm)

# # 模型诊断：检验各种模型假设是否近似成立，以及查看数据是否存在异常值；
# print(str2 + '4.2. 全模型模型诊断')
# plot_lr_diagnosis(lm, traindata, target)

# 多重共线性
print(str2 + '4.3. 多重共线性（膨胀因子)')
vif = calc_vif(traindata[xcols])
print(vif)



################# 6. 模型选择 ################
print(str1 + ' 5. 模型选择')

# aic：使aic达到最小的模型为“最优”模型
aic_lm = forward_selected(traindata[selected+ [target]], target, 'AIC')
print(str2 + '5.1. AIC')
model_report(aic_lm)

# bic：使bic达到最小的模型为“最优”模型
bic_lm = forward_selected(traindata[selected+ [target]], target, 'BIC')
print(str2 + '5.2. BIC')
model_report(bic_lm)



################# 6. 模型预测 ################
print(str1 + ' 6. 模型预测')
testdata = modeldata[modeldata.year.isin(args['split_data_args']['testyear'])]
result = {'Base': mean_squared_error(testdata[target], testdata['ROE'])**0.5
           , 'Model': mean_squared_error(testdata[target], lm.predict(testdata[xcols]))**0.5
           , 'AIC Model': mean_squared_error(testdata[target], aic_lm.predict(testdata[selected]))**0.5
           , 'BIC Model': mean_squared_error(testdata[target], bic_lm.predict(testdata[selected]))**0.5
          }
print(pd.DataFrame(data=result.values(), index=result.keys(), columns=['平均预测误差']).sort_index(ascending=False))




