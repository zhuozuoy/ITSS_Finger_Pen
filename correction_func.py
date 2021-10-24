import numpy as np
from utils import *

def viterbi(trainsition_p,emission_p,labels_cate,lambda_a=10,lambda_b=1):
    p_current = (emission_p[0]**lambda_a*trainsition_p[0]**lambda_b)
    seq = [i for i in labels_cate]
    for i in range(1,len(emission_p)):
        seq_temp = ['' for i in labels_cate]
        p_temp = np.zeros(p_current.shape)
        for j,c in enumerate(labels_cate):
            tp_c = (p_current*(trainsition_p[1:,j].T)**lambda_b*emission_p[i][j]**lambda_a)
            argmax_tpc = np.argmax(tp_c)
            seq_temp[j] = seq[argmax_tpc] + c
            p_temp[j] = np.max(tp_c)
          # print(i,c,p_temp[j],seq_temp[j])
        seq = seq_temp
        p_current = p_temp
    return p_current,seq


def straghit_forward(trainsition_p, emission_p, labels_cate, lambda_a=0.05):
    p_current = ((1 - lambda_a) * emission_p[0] + lambda_a * trainsition_p[0])
    c_index_current = np.argmax(p_current)
    seq = [labels_cate[c_index_current]]
    for i in range(1, len(emission_p)):
        p_current = ((1 - lambda_a) * emission_p[i] + lambda_a * trainsition_p[c_index_current])
        c_index_current = np.argmax(p_current)
        seq.append(labels_cate[c_index_current])
    return seq


def correction_result(p_emission,lambda_a=10,lambda_b=1):
    labels_cate = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    bichargram_freq = pk.load(open('./bichargram_freq.pkl','rb'))
    p_trainsition = get_trainsition_p(bichargram_freq).values

    p_current,seq = viterbi(p_trainsition,p_emission,labels_cate,lambda_a=lambda_a,lambda_b=lambda_b)
    result = dict(zip(seq, p_current))
    result = sorted(result.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return result


def word_correction(best_word,p_emission,lambda_a=10,lambda_b=1):

    word2p = pk.load(open('./word2p.pkl', 'rb'))
    labels_cate = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    label2id = dict(zip(labels_cate, [i for i in range(len(labels_cate))]))
    candidates = []
    for i,e in enumerate(best_word):
        for j,ee in enumerate(best_word):
            if j<i:
                continue
            elif i==j:
                for ci in labels_cate:
                    temp = best_word[:i]+ci+best_word[i+1:]
                    if temp in word2p:
                        candidates.append(temp)
            else:
                for ci in labels_cate:
                    for cj in labels_cate:
                        temp = best_word[:i]+ci+best_word[i+1:j]+cj+best_word[j+1:]
                        if temp in word2p:
                            candidates.append(temp)
    candidates2p = {}
    if best_word not in word2p:
        p = 3.0293079247958117e-10/2
        for i,c in enumerate(best_word):
            p *= p_emission[i][label2id[c]]**lambda_a
        candidates2p[best_word] = p
    else:
        p = word2p[best_word]**lambda_b
        for i,c in enumerate(best_word):
            p *= p_emission[i][label2id[c]]**lambda_a
        candidates2p[best_word] = p

    for cdd in candidates:
        p = word2p[cdd]**lambda_b
        for i,c in enumerate(cdd):
            p *= p_emission[i][label2id[c]]**lambda_a
        candidates2p[cdd] = p

    result = sorted(candidates2p.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return result

if __name__ == '__main__':
    # p_emission = simulate_correction_p()
    # result = correction_result(p_emission)
    # print(result)
    # best_word = result[0][0]
    # word_correction = word_correction(best_word,p_emission)
    # print(word_correction)
    word2p = pk.load(open('./word2p.pkl', 'rb'))
    print(word2p['words'])
    # result = sorted(word2p.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
    # print(result[0])



