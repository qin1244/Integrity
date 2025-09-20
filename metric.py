import numpy as np
import torch
from rouge_score import rouge_scorer

###### 导入ROUGE评估函数计算ROUGE-L指标
rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

### 根据GT答案及生成回答计算回答的Rouge Score
def getRouge(rouge, generations, answers):
    # results = rouge.compute(predictions=[generations], references=[answers], use_aggregator=False)
    results = rouge.score(target = answers, prediction = generations)
    RoughL = results["rougeL"].fmeasure  #fmeasure/recall/precision
    return RoughL


#### 根据多次输出计算输出不同sentence的predictive entropy
### batch_scores (array(logits), array(logits)), 长度为输出句子个数
### logits的维度为num_seq x vocab_size
### num_tokens list类型，每个seq token的个数
def get_lenghthNormalized_entropy(batch_scores, num_tokens):  
    seq_entropy = np.zeros(len(num_tokens))  ## 保存每个seq的log(p)乘积
    for ind1, logits in enumerate(batch_scores): 
        for ind2, seq_logits in enumerate(logits):
            if ind1 < num_tokens[ind2]:
                conf, _ = torch.max(seq_logits.softmax(0), dim=0)
                seq_entropy[ind2] = seq_entropy[ind2] + np.log(conf.cpu().numpy())
    normalized_entropy = 0
    for ind, entropy in enumerate(seq_entropy):
        normalized_entropy += entropy/num_tokens[ind]
    normalized_entropy = -1.0* normalized_entropy/len(num_tokens)
    return normalized_entropy


###### 计算最后一个token特征的语义散度的作为句子的语义散度
###### 需要考虑每个句子的长度不一致，去除padding的token的影响
###### hidden_states : (num_tokens, num_layers, num_seq, num_input_tokens/1, embedding_size)
def getEigenIndicator(hidden_states, num_tokens):
    alpha = 1e-3
    selected_layer = int(len(hidden_states[0])/2)
    # selected_layer = -1
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for ind in range(hidden_states[1][-1].shape[0]):
        last_embeddings[ind,:] = hidden_states[num_tokens[ind]-1][selected_layer][ind,0,:]
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s


########## 根据输出的多个回复计算Lexical Similarity
###### generated_texts 为list类型, 输出的文本字符串, 维度为num_seq
def getLexicalSim(generated_texts):
    LexicalSim = 0
    for i in range(len(generated_texts)):
        for j in range(len(generated_texts)):
            if j<=i:
                continue
            LexicalSim += getRouge(rougeEvaluator, generated_texts[i], generated_texts[j])
    LexicalSim = LexicalSim/(len(generated_texts)*(len(generated_texts)-1)/2)
    return LexicalSim

#没有EOStoken可能会溢出
def get_num_tokens(generation):  # generation: num_seq x max(num_tokens)
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2 and id<128000:
                count+=1
        if count==len(ids):
         num_tokens.append(count)
        else:
            num_tokens.append(count+1)
    return num_tokens



    







