from asyncio.windows_events import NULL
from cmath import log, log10
from math import log2
from torch import double, float64
import wikipedia as wk
import requests
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
def getPage(title):
    wk.set_lang('en')
    timeStart = time.time()
    page = wk.page(title)

    timeEnd = time.time()
    sec = timeEnd - timeStart
    result_list = str(datetime.timedelta(seconds=sec))
    print(result_list)
    if(page):
        #print(page.content)
        return page
    return NULL
def calcEntrophy(allBacklinksNum, asAnchortextNum):#mention A 에 대한 entrophy를 구한다
    #allBacklinksNum = A로 인해 발생한 concept후보들 각각의 전체 백링크 수
    #asAnchortextNum = 각각의 백링크 중에 해당하는 페이지로 연결된 링크의 anchor text가 A인 백링크 수

    length = len(allBacklinksNum)#길이는 같은걸로 간주한다
    sum=0
    for i in range(length):
        if(asAnchortextNum[i] == 0 or allBacklinksNum[i] == 0):#둘 중 하나라도 0이면 넘김
            continue
        temp = asAnchortextNum[i]/allBacklinksNum[i]
        sum -= temp * log10(temp)

    return sum
def calcMentionToConceptTP():#mention vertex에서 concept vertex로 가는 간선의 가중치
    return

def calcPR0(pagesNum, asAnchorTextPagesNum):#mention vertex의 PR0를 계산하는 함수로 concept vertex의 PR0는 따로 처리해줄것
    #PagesNum = 분모가 될 숫자들의 리스트(배열) 해당 phrase를 가지고있는 페이지 수
    #AsAnchorText = 분자가 될 숫자들의 리스트(배열) 해당 phrase를 anchor text로 가지고 있는 페이지 수
    sum = 0 
    length = len(pagesNum)#pagesNum과 asAnchorText의 길이는 같은걸로 간주한다
    PR0 = []
    for i in range(length):#집합의 크기만큼 반복
        #z를 제외한 결과를 저장
        PR0.append(asAnchorTextPagesNum[i]/pagesNum[i])
        sum += PR0[i]

    #mention vertex와 concept vertex 모두 합해서 계산하게 되면 z의 값이 바뀌게되는데 구분하기 힘들것 같아서
    #우선은 mention vertex만 모두 합한 값으로 z를 구하는걸로 했음 문제가 있으면 나중에 수정하면 될듯

    z = 1 / sum / length #z에 관한 수식 풀어내면 이런 식이 나옴
    
    for i in range(length):#집합의 크기만큼 반복
        PR0[i] *= z#계산한 z를 곱해줘서 계산을 마친다

    #PR0 계산 결과는 리스트로 리턴
    return PR0

def PR0():#PR0를 구하기 위해선 각 vertex에 연결된 간선들과 간선들의 가중치, 인접한 vertex에대한 데이터가 필요함
    #자료 vertex와 edge들에 대한 자료구조를 명확히 만들고나서 작성해야할듯
    return 


def test ():
    entrophy = []
    entrophy1 = []
    entrophy2 = []
    entireBacklinkNum = 2000
    allBacklinkNum = 150
    candidateConceptNum = 50
    for j in range(1,allBacklinkNum):
        tt = j/allBacklinkNum
        entrophy.append(-tt * log10(tt))
        
    for j in range(1,allBacklinkNum):
        tt = j/candidateConceptNum
        entrophy1.append(-tt * log10(tt))

    for j in range(1,allBacklinkNum):
        tt = j/entireBacklinkNum
        entrophy2.append(-tt * log10(tt))

    plt.plot(range(1,allBacklinkNum),entrophy,'r^-')
    plt.plot(range(1,allBacklinkNum),entrophy1,'b^-')
    plt.plot(range(1,allBacklinkNum),entrophy2,'g^-')
    plt.show()
'''
pn = [13201,845,35756,12103,1204]
apn = [350,20,541,124,44]

pr0 = calcPR0(pn,apn)
print(pr0)
test()
'''