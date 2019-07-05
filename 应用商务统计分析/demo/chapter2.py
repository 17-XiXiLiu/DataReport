import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# print配置
str1 = '\n'*4 + '#'*80 
str2 = '\n'*2 + '-'*80 
str3 = '\n'*1 + '-'*40

plt.rcParams['font.sans-serif']=['SimHei'] 


def get_data(path, index='name'):
    df = pd.read_csv(open(path, 'r', encoding="utf8"))
    df.set_index('name', inplace=True)
    return df


def get_key(Dict, value):
    return [k for k, v in Dict.items() if value in v or value == v]


def clean_data(df, y, xdict):
    '''
    数据清洗
    1.数据类型转换；
    2.有效样本和变量选取；
    3.因变量对数转换；
    '''
    print(str2 + 'Begin')
    print(df.shape)
    xcols = list(xdict.keys())
    
    # 数据类型转换
    for col in xcols: 
        df[col] = df[col].astype(str)
        print('\n %s: %d' % (xdict[col], len(df[col].unique())))
        print(df[col].unique())
        
    # 有效样本集
    df['city'] = df.location0.map(lambda x: get_key(city_dict, x)[0])
    df.type = df.type.map(lambda x: '商业' if '商业' in x else x)
    df = df.loc[(df.price_unit=='元/平(均价)') & (df.sale_status=='在售') & (df.city=='广州') & (df.room_num != '0'), xcols + [y]]  
    
    # 转换因变量
    df[y] = df[y].astype(float)
    df[y + '_log'] = np.log(df[y]) #对数转换的作用：https://www.zhihu.com/question/22012482
    print(str2 + 'End')
    print(df.shape)

    return df


def desc_data(df, y, xdict):
    '''
    描述性分析
    1.因变量分布图和基本统计量；
    2.自变量箱装图和基本统计量；
    '''
    df[y].hist(figsize=(10, 5))
    plt.title('因变量分布', fontsize=15, fontweight='bold')
    plt.show()
    print(df[y].describe())
    
    for col in xdict.keys():
            df.boxplot(by=col, column=y, grid=False)
            plt.title('%s' % xdict[col], fontsize=15, fontweight='bold')
            plt.suptitle('')
            plt.xlabel('')
            plt.ylabel(y)
            plt.show()
            print('\n %s: %d' % (xdict[col], len(df[col].unique())))
            print(df[col].value_counts())
   

def bonferroni(totAlpha, pvalues):
    '''
    bomferroni方法：将总体错误概率控制在alpha水平下
    '''
    n = len(pvalues)
    subAlpha = totAlpha / n
    valid = [ i for i in pvalues.index if pvalues[i] < subAlpha ]
    return valid


def anova(df, target, xcols, formula=None, interaction=False):
    '''
    可以理解为，对离散变量进行哑变量转换，再建立简单线性模型，中intercept则为共线性中drop掉的那一列
    若某项不显著，则说明对这类数据预测效果不好
    '''
    if formula == None:
        if interaction:
            formula = target +  '~ ' + '*'.join(map(lambda x: 'C(%s)'% x, xcols))
        else:
            formula = target +  '~ ' + '+'.join(map(lambda x: 'C(%s)'% x, xcols))
    lm = ols(formula, df).fit()
    anova_res = anova_lm(lm, type=3)
    intercept = [i for i in np.unique(df[xcols]) if i not in str(lm.params.index)]
    print(str2)
    print(anova_res)
    print(str3 + 'Intercept is %s' % intercept)
    print(lm.summary().tables[1])
    print(str3 + 'Bonferroni\n' , bonferroni(0.1, lm.pvalues))   
    return lm



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
    fig = plt.figure(figsize = (8, 6), dpi = 100)

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
xdict = {  'location0': '城区'
         , 'room_num': '户型种类数量'
         , 'type':'物业类型'
         , 'tag_num': '标签数量'  
#          , 'location1': '地理位置'
#          , 'city':'城市' 
      }
y = 'price' # '平均房价/元'           

# 其他参数
filepath =  u'houseprice_gz.csv' 
city_dict = {'广州': ['增城','花都','从化','南沙','荔湾','番禺','黄埔','天河','白云','海珠','越秀']
             , '佛山': ['禅城','顺德','高明','南海','三水']
             , '清远': ['清新区','清城区','英德市']      
            } # 城市

    
    
################ 1. 数据采集 #################
print(str1 + ' 1. 数据采集')
df = get_data(filepath)



################ 2. 数据清洗 #################
print(str1 + ' 2. 数据清洗')
modelDf = clean_data(df.copy(), y, xdict)


################ 3. 描述性分析 #################
print(str1 + ' 3. 描述性分析')

print(str2 + ' 3.1 因变量（原始）')
desc_data(modelDf, y, xdict)    

print(str2 + ' 3.2 因变量（对数转换）')
desc_data(modelDf, y + '_log', xdict)  



################ 4. 方差分析 ################
print(str1 + ' 4. 方差分析')
target = y + '_log'

# # 4.1. 单因素
# print(str2 + ' 4.1. 单因素')
# for col in xdict.keys():
#     print(str2 + xdict[col])
#     lm = anova(modelDf, target, [col])


# # 4.2. 双因素
# print(str2 + ' 4.2. 双因素')
# couples = [[list(xdict.keys())[i], list(xdict.keys())[j]] for i in range(len(xdict)) for j in range(i+1, len(xdict))]

# # 4.2.1. 简单叠加
# print(str3 + ' 4.2.1. 双因素（简单叠加）')
# for couple in couples:
#     print(str2 + xdict[couple[0]] + ' & ' + xdict[couple[1]])
#     lm = anova(modelDf, target, couple)
#     print(lm.summary().tables[1])
    
# # 4.2.2. 交互作用
# print(str3 + ' 4.2.2. 双因素（交互作用）')
# for couple in couples:
#     print(str2 + xdict[couple[0]] + ' & ' + xdict[couple[1]])
#     lm = anova(modelDf, target, couple, interaction=True)
#     print(lm.summary().tables[1])


# # 4.3. 多因素
# print(str2 + ' 4.3. 多因素')
# print(str2 + 'ALL of ' + str(list(xdict.values())))
# lm =  anova(modelDf, target, list(xdict.keys()))
# # #（考虑个别交互的多因素方差分析）
# # couple = ['location0', 'type']
# # formula = target +  '~ C({0}) * C({1}) + '.format(couple[0], couple[1]) + '+'.join(map(lambda x: 'C(%s)'% x, xdict.keys())) #考虑个别交互（城区和住房类型交互）
# # lm =  anova(train, target, list(xdict.keys()), formula=formula)
# print(lm.summary().tables[1])



################ 5. 数据建模 ################
print(str1 + ' 5. 数据建模')


# 训练模型
lm =  anova(modelDf, target, list(xdict.keys()))

# 模型诊断：检验各种模型假设是否近似成立，以及查看数据是否存在异常值；
print(str2 + '全模型模型诊断')
plot_lr_diagnosis(lm, modelDf, target)



