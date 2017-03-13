import os
from collections import Counter

def read_data(fname, count, word2idx):
    if os.path.isfile(fname): # 파일에서 모든 라인을 읽어온다
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise("[!] Data %s not found" % fname)

    words = []
    for line in lines: # 단어들을 따로 리스트로 만들어 담는다
        words.extend(line.split())

    if len(count) == 0:
        count.append(['<eos>', 0])

    count[0][1] += len(lines)
    count.extend(Counter(words).most_common())
    # 각 토큰 당 빈도를 세는 리스트인듯?
    # stop word로 사용할 생각인가 .most_common으로 보아하니...
    
    # 각 word를 index로 매핑하는 dict
    if len(word2idx) == 0:
        word2idx['<eos>'] = 0

    for word, _ in count:
        # count로부터 토큰을 모두 word2inx로 매핑
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    
    ### 각각 다른 데이터를 읽더라도
    ### 이런식으로 먼저 count와 word2inx를 추가해주고
    ### 그 다음부터 각 라인(문장)에 대해 index로 변환한다
    
    data = list()
    for line in lines:
        for word in line.split():
            index = word2idx[word]
            data.append(index)
        data.append(word2idx['<eos>'])
    
    # [[문장],[문장],...] -> [indx,indx,indx, 0<eos>,idx,idx,..,...]
    
    print("Read %s words from %s" % (len(data), fname))
    return data
