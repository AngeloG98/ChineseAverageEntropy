import os
import math
import jieba
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
mpl.rcParams['font.sans-serif'] = ['SimHei']


"""
DefineClass：
    1.存储数据
    2.短句
    3.分词
    4.计算N元模型词频
    5.计算N元模型平均信息熵
"""
class ChineseData:
    def __init__(self, txtname='', txt='', sentences=[], words=[], entropyinit = {}):
        self.txtname = txtname
        self.txt = txt
        self.sentences = sentences
        self.words = words
        global punctuation
        self.punctuation = punctuation
        global stopwords
        self.stopwords = stopwords
        self.entropy = entropyinit

    def sepSentences(self):
        line = ''
        sentences = []
        for w in self.txt:
            if w in self.punctuation and line != '\n':
                if line.strip() != '':
                    sentences.append(line.strip())
                    line = ''
            elif w not in self.punctuation:
                line += w
        self.sentences = sentences

    def sepWords(self):
        words = []
        dete_stopwords = 0
        if dete_stopwords:
            for i in range(len(self.sentences)):
                words.extend([x for x in jieba.cut(self.sentences[i]) if x not in self.stopwords])
        else:
            for i in range(len(self.sentences)):
                words.extend([x for x in jieba.cut(self.sentences[i])])
        self.words = words
    
    def getNmodel(self, phrase_model, n):
        if n == 1:
            for i in range(len(self.words)):
                phrase_model[self.words[i]] = phrase_model.get(self.words[i], 0) + 1
        else:
            for i in range(len(self.words) - (n - 1)):
                if n == 2:
                    condition_t = self.words[i]
                else:
                    condition = []
                    for j in range(n-1):
                        condition.append(self.words[i + j])
                    condition_t = tuple(condition)
                phrase_model[(condition_t, self.words[i+n-1])] = phrase_model.get((condition_t, self.words[i+n-1]), 0) + 1
    
    def getN_1model(self, phrase_model, n):
        if n == 1:
            for i in range(len(self.words)):
                phrase_model[self.words[i]] = phrase_model.get(self.words[i], 0) + 1
        else:
            for i in range(len(self.words) - (n - 1)):
                condition = []
                for j in range(n):
                    condition.append(self.words[i + j])
                condition_t = tuple(condition)
                phrase_model[condition_t] = phrase_model.get(condition_t, 0) + 1
                                
    def calcuNmodelEntropy(self, n, entropy_dic):
        if n < 1 or n >= len(self.words):
            print("Wrong N!")
        elif n == 1:
            phrase_model = {}
            self.getNmodel(phrase_model, 1)
            model_lenth = len(self.words)
            entropy_dic[n] = sum([-(phrase[1] / model_lenth) * math.log(phrase[1] / model_lenth, 2) for phrase in phrase_model.items()])
            entropy_dic[n] = round(entropy_dic[n], 4) 
            # self.entropy[n] = sum([-(phrase[1] / model_lenth) * math.log(phrase[1] / model_lenth, 2) for phrase in phrase_model.items()])
            # self.entropy[n] = round(self.entropy[n], 4)  
            # self.entropy.append(sum([-(phrase[1] / model_lenth) * math.log(phrase[1] / model_lenth, 2) for phrase in phrase_model.items()]))         
            # self.entropy = sum([-(phrase[1] / model_lenth) * math.log(phrase[1] / model_lenth, 2) for phrase in phrase_model.items()])
        else:
            phrase_model_n = {}
            phrase_model_n_1 = {}
            self.getNmodel(phrase_model_n, n)
            self.getN_1model(phrase_model_n_1, n - 1)
            phrase_n_len = sum([phrase[1] for phrase in phrase_model_n.items()])
            entropy = []
            for n_phrase in phrase_model_n.items():
                p_xy = n_phrase[1] / phrase_n_len
                p_x_y = n_phrase[1] / phrase_model_n_1[n_phrase[0][0]] 
                entropy.append(-p_xy * math.log(p_x_y, 2))
            entropy_dic[n] = round(sum(entropy), 4)
            # self.entropy[n] = round(sum(entropy), 4)
            # self.entropy.append(round(sum(entropy), 4))
            # self.entropy = round(sum(entropy), 4)
 
    def run(self):
        self.sepSentences()
        self.sepWords()
        entropy_dic = {}
        self.calcuNmodelEntropy(1, entropy_dic)
        self.calcuNmodelEntropy(2, entropy_dic)
        self.calcuNmodelEntropy(3, entropy_dic)
        self.calcuNmodelEntropy(4, entropy_dic)
        self.entropy = entropy_dic
        

"""
InputOutput：
    1.读取文章数据
    2.读取标点
    3.读取停词（未使用）
    4.输出图表结果
"""
def read_all_files(path):
    data_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            with open(filename, 'r', encoding='ANSI') as f:
                txt = f.read()
                txt = txt.replace('本书来自www.cr173.com免费txt小说下载站', '')
                txt = txt.replace('更多更新免费电子书请关注www.cr173.com', '')
                d = ChineseData()
                d.txt = txt
                d.txtname = file.split('.')[0]
                data_list.append(d)
            f.close()
    return data_list

def read_punctuation_list(path):
    punctuation = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    punctuation.extend(['\n', '\u3000', '\u0020', '\u00A0'])
    return punctuation

def read_stopwords_list(path):
    stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    return stopwords

def draw_results_sub(data):
    k = len(data)
    num = []
    for i in range(k):
        count = 0
        for j in range(len(data[i].sentences)):
            count += len(data[i].sentences[j])
        num.append(count)
    labels = []
    for i in range(k):
        labels.append(data[i].txtname + '\n'  + str(num[i]))
    entropylist = []
    for i in range(4):
        entropy = []
        for j in range(k):
            entropy.append(data[j].entropy[i + 1])
        entropylist.append(entropy)

    fonten = {'family': 'Times New Roman', 'size': 10}
    
    width = 0.3
    ind = np.linspace(0.5, 0.5 + (k-1) , 1 + (k-1))
    fig = plt.figure(1, figsize=(20, 10), dpi=300)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.bar(ind, entropylist[0], width, color='green')
    ax2.bar(ind, entropylist[1], width, color='blue')
    ax3.bar(ind, entropylist[2], width, color='yellow')
    ax4.bar(ind, entropylist[3], width, color='red')

    ax1.set_xticks(ind)
    ax1.set_xticklabels(labels, size='small', rotation=40, fontweight='bold')
    ax1.set_ylim(min(entropylist[0]) - 0.1, max(entropylist[0]) + 0.1)
    ax1.set_ylabel('1-gram Average Entropy', fontdict=fonten, fontweight='bold')
    for a, b in zip(ind, entropylist[0]):
        ax1.text(a, b + 0.01, b, ha='center', va='bottom', fontdict=fonten, fontsize=7)
        
    ax2.set_xticks(ind)
    ax2.set_xticklabels(labels, size='small', rotation=40, fontweight='bold')
    ax2.set_ylim(min(entropylist[1]) - 0.1, max(entropylist[1]) + 0.1)
    ax2.set_ylabel('2-gram Average Entropy', fontdict=fonten, fontweight='bold')
    for a, b in zip(ind, entropylist[1]):
        ax2.text(a, b + 0.01, b, ha='center', va='bottom', fontdict=fonten, fontsize=7)

    ax3.set_xticks(ind)
    ax3.set_xticklabels(labels, size='small', rotation=40, fontweight='bold')
    ax3.set_ylim(min(entropylist[2]) - 0.1, max(entropylist[2]) + 0.1)
    ax3.set_ylabel('3-gram Average Entropy', fontdict=fonten, fontweight='bold')
    for a, b in zip(ind, entropylist[2]):
        ax3.text(a, b + 0.01, b, ha='center', va='bottom', fontdict=fonten, fontsize=7)

    ax4.set_xticks(ind)
    ax4.set_xticklabels(labels, size='small', rotation=40, fontweight='bold')
    ax4.set_ylim(max(min(entropylist[3]) - 0.1, 0), max(entropylist[3]) + 0.1)
    ax4.set_ylabel('4-gram Average Entropy', fontdict=fonten, fontweight='bold')
    for a, b in zip(ind, entropylist[3]):
        ax4.text(a, b + 0.01, b, ha='center', va='bottom', fontdict=fonten, fontsize=7)
    
    fig.suptitle('Chinese Average Entropy of Louis Cha Novels', bbox={'facecolor': '0.8', 'pad': 7}, fontdict=fonten)
    plt.savefig('chineseaverageentropy.png', bbox_inches='tight')
    # plt.show()
    plt.close()


"""
Main：
"""
if __name__ == "__main__":
    data_dir_path = '.\\DatabaseChinese'
    stopwords_path = '.\\StopWord\\cn_stopwords.txt'
    punctuation_path = '.\\StopWord\\cn_punctuation.txt'

    global stopwords
    stopwords = read_stopwords_list(stopwords_path)
    global punctuation
    punctuation = read_punctuation_list(punctuation_path)
    data_list = read_all_files(data_dir_path)

    for i in range(len(data_list)):
        logging.info('正在处理《'+data_list[i].txtname+'》...')
        ChineseData.run(data_list[i])
        logging.info('《'+data_list[i].txtname+'》处理完成...')

    np.save('data_list.npy', data_list)
    # data_list=np.load('data_list.npy', allow_pickle=True)
    
    draw_results_sub(data_list)
    
