# -*- encoding=utf-8 -*-
import pandas as pd
import math
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

origin_data = pd.read_excel("data.xlsx", encoding='utf8')
data = origin_data.iloc[:, 1:10]

def get_entropy_weight(data):
    """
    :param data: 评价指标数据框
    :return: 各指标权重列表
    """
    data = (data - data.min())/(data.max() - data.min())
    m,n=data.shape
    k=math.log(1/float(m))
    yij=data.sum()
    pij=data/yij
    pij_=data/yij
    pij_[pij_==0]=0.0001
    lnp=pij_.applymap(lambda x: math.log(x))
    eij=-k*(pij*lnp).sum()
    wij=(1-eij)/(1-eij).sum()
    return  wij

wij=get_entropy_weight(data)
score=(wij*data).sum(axis=1)
origin_data['score']=score
result = origin_data.sort_values(by='score', axis=0, ascending=False)
result['rank'] = range(1, len(result) + 1)
result.to_excel('result.xlsx',index=False)
