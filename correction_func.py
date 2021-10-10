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


if __name__ == '__main__':
    labels_cate = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    bichargram_freq = pk.load(open('./bichargram_freq.pkl','rb'))
    softmax_smooth_t = simulate_correction_p()
    p_x = get_trainsition_p(bichargram_freq).values

    p_current,seq = viterbi(p_x,softmax_smooth_t,labels_cate)
    result = dict(zip(seq, p_current))
    result = sorted(result.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print(result)

    # result = straghit_forward(p_x,softmax_smooth_t,labels_cate)
    # print(result)
