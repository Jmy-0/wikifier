



from cmath import log10


class Vertex:
    def __init__(self, type, name):
        self.type = type
        self.name = name
        self.PR0 = 0
        self.PR = -1
        self.edges = []

class Edge:
    #global EntireSR
    EntireSR = 0
    def __init__(self, type):
        self.P = -1
        self.SR = -1
        self.type = type#mention to concept(0) or concept to concept(1)
        print("type = %d"%(self.type))
    @classmethod
    def conceptToConcept(cls,start_set:list,end_set:list, N:int):
        temp = Edge(1)
        temp.calcSR(start_set,end_set,N)
        return temp

        
    def calcMtoC(self,a:int,c:int):
        self.P = c/a

    def calcCtoC(self,set1,set2):#전체 concept 정점에 대한 SR이 필요하다
        
        return
    def calcSR(self,start_set, end_set, N):
        sameNum = 0
        for i in start_set:
            for j in end_set:
                if(i == j):
                    sameNum +=1
        
        startLen = len(start_set)
        endLen = len(end_set)

        denominator = (log10(N) - log10(min(startLen,endLen)))#분모
        numerator = (log10(max(startLen,endLen)) - log10(sameNum)) #분자


        self.SR = 1- numerator / denominator
        Edge.EntireSR += self.SR
        print(Edge.EntireSR)
        return

    def printing(self):
        print(self.type)

li = ['10','5']
li2 = ['101','5']
w=Edge.conceptToConcept(li,li2,20)
w1=Edge(0)
w2=Edge(0)
