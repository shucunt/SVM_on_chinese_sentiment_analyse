# -*- coding: utf-8 -*-

import os
import random
import chardet
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
# get weibo_data from xml
def opt_xml(file_src):
    tree = ET.ElementTree(file=file_src)
    root = tree.getroot()
    data = []
    lable = []
    #print root.tag, root.attrib
    for child_of_root in root:
        for grand_child in child_of_root:
            if grand_child.attrib.has_key('polarity'):
                data.append(grand_child.text)
                lable.append(grand_child.attrib['polarity'])
                #print grand_child.text, grand_child.attrib['polarity']
    #print len(data)
    return data, lable

#get data from book_review's txt
def opt_book_review_data(file_src, domain):
    file_list = os.listdir(os.path.join(file_src, domain))
    #print file_list
    data = []
    lable = []
    for file in file_list:
        file_read = open(os.path.join(os.path.join(file_src, domain), file), 'r')
        try:
            #the code of data is gb2312
            l = file_read.readlines()[1].decode('gb2312', 'ignore')
            #print chardet.detect(l)
            data.append(l)
            if domain == 'neg':
                lable.append('NEG')
            else:
                lable.append('POS')
        finally:
            file_read.close()
    
    #print len(data), len(lable)
    #print data[0]
    return data, lable




#write processed data to file
def write_to_file(data, lable, file_src, domain):
    file_write = open(os.path.join(file_src, domain + '_data'), 'w')
    try:
        for i in range(len(data)):
            file_write.write(data[i].encode('gbk') + '\n')
    finally:
        file_write.close()
    file_write = open(os.path.join(file_src, domain + '_lable'), 'w')
    try:
        for i in range(len(lable)):
            file_write.write(lable[i] + '\n')
    finally:
        file_write.close()


#mix neg_data and pos_data
def mix_data(data1, lable1, data2, lable2):
    for i in range(len(data2)):
        temp_num = random.randint(0, len(data1))
        #print temp_num
        data1.insert( temp_num, data2[i])
        lable1.insert(temp_num, lable2[i])
    return data1, lable1

#get sentiment word from txt
def read_sentiment_word(file_src_pos, file_src_neg):
    file_read = open(file_src_pos, 'r')
    try:
        l_pos = file_read.readlines()
    finally:
        file_read.close()
    file_read = open(file_src_neg, 'r')
    try:
        l_neg = file_read.readlines()
    finally:
        file_read.close()
    l = []
    for i in range(len(l_pos)):
        if len(l_pos[i]) != 0:
            l.append(l_pos[i].strip())
    for i in range(len(l_neg)):
        if len(l_neg[i]) != 0:
            l.append(l_neg[i].strip())
    return l

#find sentiment word exist in train_data
def exit_sentiment_word(all_sentiment_word, data, file_src):
    exit_lable = [0 for i in range(len(all_sentiment_word))]
    for sentc in data:
        for i in range(len(all_sentiment_word)):
            if all_sentiment_word[i] in sentc:
                exit_lable[i] = 1
    
    exit_sentiment_words = []
    for i in range(len(all_sentiment_word)):
        if exit_lable[i] == 1:
            #print all_sentiment_word[i]
            exit_sentiment_words.append(all_sentiment_word[i])
    print 'the dimension of the vector ' + str(len(exit_sentiment_words))
    #for i in range(len(exit_sentiment_words)):
        #print exit_sentiment_words[i]
    file_write = open(file_src, 'w')
    try:
        for ele in exit_sentiment_words:
            file_write.write(ele + '\n')
    finally :
        file_write.close()
    return exit_sentiment_words

#return the vector of sentence
def sentc_embedding(data, exit_sentiment_words):
    sentc_embedding_result = []
    for ele in data:
        temp = [0 for i in range(len(exit_sentiment_words))]
        for i in range(len(exit_sentiment_words)):
            if ele.count(exit_sentiment_words[i]) < 10:
                temp[i] = ele.count(exit_sentiment_words[i])
        sentc_embedding_result.append(temp)
    return sentc_embedding_result

#operate the lable,turn NEG to -1  ;turn POS to 1
def opt_lable(lable):
    opt_lable_result = []
    for ele in lable:
        if 'NEG' in ele:
            opt_lable_result.append(-1)
        else:
            opt_lable_result.append(1)
    return opt_lable_result

#write the vector of sentence and lable to file
def write_embedding(data, lable, file_src):
    file_src_data = os.path.join(file_src, 'data_embedding')
    file_src_lable = os.path.join(file_src, 'lable_embedding')
    file_write = open(file_src_data, 'w')
    try:
        for ele in data:
            file_write.write(' '.join(str(num) for num in ele) + '\n')
    finally:
        file_write.close()
    file_write = open(file_src_lable, 'w')
    try:
        for ele in lable:
            file_write.write(str(ele) + '\n')
    finally:
        file_write.close()


#print the information of data
def info_about_data(data, lable):
    neg_num = 0
    for ele in lable:
        if ele < 0:
            neg_num += 1
    print 'the amount of all data ' + str(len(lable))
    print 'the proportion of negative data ' + str(neg_num * 1.0 / len(lable))


    
#predict the lable of predict_data
def run_predict_data(predict_data):
    '''
    #this code is for weibo-data
    huirong_data, huirong_lable = opt_xml(os.path.join(os.path.abspath('..'), 'data', 'weibo_sentiment', 'ori_data', 'labelled dataset', 'hui_rong_an.xml'))
    ipad_data, ipad_lable = opt_xml(os.path.join(os.path.abspath('..'), 'data', 'weibo_sentiment', 'ori_data', 'labelled dataset', 'ipad.xml'))
    data, lable = mix_data(huirong_data, huirong_lable, ipad_data, ipad_lable)
    file_src_pos = os.path.join(os.path.abspath('..'), 'data', 'ntusd_sectiment_dict', 'ntusd-positive.txt')
    file_src_neg = os.path.join(os.path.abspath('..'), 'data', 'ntusd_sectiment_dict', 'ntusd-negative.txt')
    all_sentiment_word = read_sentiment_word(file_src_pos, file_src_neg)
    exit_sentiment_words = exit_sentiment_word(all_sentiment_word, data)
    '''
    #this code is for book_review data,and get the sentiment word from file
    file_read = open(os.path.join(os.path.abspath('..'), 'data', 'chnsenticorp', 'Dangdang_Book_4000', 'senti_word'))
    try:
        exit_sentiment_words = file_read.readlines()
    finally:
        file_read.close()
    for i in range(len(exit_sentiment_words)):
        exit_sentiment_words[i] = exit_sentiment_words[i].strip()
    #print len(exit_sentiment_words)

    predictX = []
    predictX.append(predict_data)
    predictX_result = sentc_embedding(predictX, exit_sentiment_words)
    return predictX_result[0]

#train the model
def run_train_data():
    '''  #weibo_data
    huirong_data, huirong_lable = opt_xml(os.path.join(os.path.abspath('..'), 'data', 'weibo_sentiment', 'ori_data', 'labelled dataset', 'hui_rong_an.xml'))
    ipad_data, ipad_lable = opt_xml(os.path.join(os.path.abspath('..'), 'data', 'weibo_sentiment', 'ori_data', 'labelled dataset', 'ipad.xml'))
    #write_to_file(huirong_data, huirong_lable, os.path.join(os.path.abspath('..'), 'data', 'weibo_sentiment', 'ori_data', 'labelled dataset'), 'huirong')
    #write_to_file(ipad_data, ipad_lable, os.path.join(os.path.abspath('..'), 'data', 'weibo_sentiment', 'ori_data', 'labelled dataset'), 'ipad')
    '''
    #book_review_data
    file_src = os.path.join(os.path.abspath('..'), 'data', 'chnsenticorp', 'Dangdang_Book_4000')
    neg_data, neg_lable = opt_book_review_data(file_src, 'neg')
    pos_data, pos_lable = opt_book_review_data(file_src, 'pos')
    data, lable = mix_data(pos_data, pos_lable, neg_data, neg_lable)
    #write_to_file(data, lable, os.path.join(os.path.abspath('..'), 'data', 'weibo_sentiment', 'ori_data', 'labelled dataset'), 'all')
    '''
    file_read = open(os.path.join(os.path.join(os.path.abspath('..'), 'data', 'weibo_sentiment', 'ori_data', 'labelled dataset', 'all_data')), 'r')
    try:
        data = file_read.readlines()
    finally:
        file_read.close()
    '''
    file_src_pos = os.path.join(os.path.abspath('..'), 'data', 'ntusd_sectiment_dict', 'ntusd-positive.txt')
    file_src_neg = os.path.join(os.path.abspath('..'), 'data', 'ntusd_sectiment_dict', 'ntusd-negative.txt')
    all_sentiment_word = read_sentiment_word(file_src_pos, file_src_neg)
    #print type(data[0])
    #print chardet.detect(data[0])
    #print all_sentiment_word[0] + '!!!' + data[0]
    file_src_senti_word = os.path.join(os.path.abspath('..'), 'data', 'chnsenticorp', 'Dangdang_Book_4000', 'senti_word')
    exit_sentiment_words = exit_sentiment_word(all_sentiment_word, data, file_src_senti_word)
    #print exit_sentiment_words[0]

    sentc_embedding_result = sentc_embedding(data, exit_sentiment_words)
    #for ele in sentc_embedding_result:
        #print ele

    opt_lable_result = opt_lable(lable)
    #print opt_lable_result
    file_src = os.path.join(os.path.abspath('..'), 'data', 'weibo_sentiment', 'ori_data', 'labelled dataset')
    #write_embedding(sentc_embedding_result, opt_lable_result, file_src)

    info_about_data(sentc_embedding_result, opt_lable_result)
    return sentc_embedding_result, opt_lable_result

if __name__ == '__main__':
    print 'no program is running!'
    #run_train_data()
    #file_src = os.path.join(os.path.abspath('..'), 'data', 'chnsenticorp', 'Dangdang_Book_4000')
    #opt_bood_review_data(file_src, 'neg')



