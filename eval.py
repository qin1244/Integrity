import os
import pickle
import string
from simcse import SimCSE
import torch
import re
import numpy as np
import jsonlines
import argparse
from tqdm import tqdm, trange
from sklearn.metrics import roc_curve, auc


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument("--filename", type=str, help="Input Filename", required=True)
parser.add_argument("--evalform", type=str, default="SimCSE", choices=["SimCSE", "EigenScore", "NormalizedEntropy", "LexicalSimilarity",  "joint"], help="Evaluation form")
parser.add_argument("--threshold_sim", default=0.75, type=float, help="Threshold of similarity")
parser.add_argument("--threshold_entropy", default=0.153, type=float, help="Threshold of Length Normalized Entropy")
parser.add_argument("--threshold_eigenScore", default=-1.00, type=float, help="Threshold of eigenScore")
parser.add_argument("--threshold_LexicalSimilarity", default=0.489, type=float, help="Threshold of Lexical Similarity ")
parser.add_argument("--model", default="../princeton-nlp/sup-simcse-roberta-large", type=str, help="Smilarity Model")
args = parser.parse_args()

uncertain_list = [
"The answer is unknown.",
"The answer is uncertain.",
"The answer is unclear.",
"It is not known.",
"We do not know.",
"I don’t know.",
"I don't have the answer to that.",
"That's outside of my expertise.",
"I don’t have enough details to provide an accurate answer.",
"I don’t have the answer right now.",
"I’m not confident about this one.",
"That’s outside my knowledge scope.",
"I don’t know the exact answer to that.",
"I’m afraid I can’t answer that directly.",
"That question seems to need more knowledge.",
"I don’t have all the information to answer the question.",
"I’m not able to answer that.",
"I'm afraid this is beyond my current knowledge.",
"This question requires more specialized knowledge than I have.",
"I don’t have the expertise to answer that accurately.",
"This is outside my knowledge base.",
"I don't have the depth of knowledge to fully answer that.",
"This question seems to require more domain-specific understanding than I possess.",
"I’m not familiar enough with this topic to give you a precise answer.",
"This is a complex issue that falls outside my area of expertise.",
"I’m not qualified to give you a complete answer on this topic.",
"That requires knowledge I don’t currently have access to.",
"I don't have enough information to answer this thoroughly.",
"This is beyond my current understanding.",
"I don’t have the required knowledge in this area to provide an answer.",
"I'm not aware of any updates beyond my knowledge cutoff",
"I'm not aware of that",
"It goes beyond what I know.",
"That’s beyond my knowledge scope.",
"I need more information",
"I need more knowledge",
"I’m afraid I can’t provide an accurate response.",
"I don’t have the right background to address this question.",
"This is an area I’m not well-versed in.",
"The question requires more knowledge.",
"I don't have access to external knowledge."
]

select_device = args.device
device = torch.device(select_device if torch.cuda.is_available() else "cpu")
threshold_map = {
    "SimCSE": args.threshold_sim,
    "EigenScore": args.threshold_eigenScore,
    "NormalizedEntropy": args.threshold_entropy,
    "LexicalSimilarity": args.threshold_LexicalSimilarity,
    "joint": " ".join([
        f"SimCSE:{args.threshold_sim}",
        f"NormalizedEntropy:{args.threshold_entropy}"
    ])
}

def read_jsonl(filename):
    data_list = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list


def remove_punctuation(input_string):
    input_string = input_string.strip().lower()
    if input_string and input_string[-1] in string.punctuation:
        return input_string[:-1]
    return input_string


def cut_sentences(content):
    sentences = re.split(r"(\.|\!|\?|。|！|？|\.{6})", content)
    return sentences


def cut_sub_string(input_string, window_size=5, punctuation=".,?!"):
    input_string = input_string.strip().lower()
    if len(input_string) < 2:
        return [""]
    if input_string[-1] in punctuation:
        input_string = input_string[:-1]
    string_list = input_string.split()
    length = len(string_list)
    if length <= window_size:
        return [input_string]
    else:
        res = []
        for i in range(length - window_size + 1):
            sub_string = " ".join(string_list[i: i + window_size])
            if sub_string != "" or sub_string != " ":
                res.append(sub_string)
        return res

def get_similarities(data_list,model_name):
    model = SimCSE(args.model)
    length = len(data_list)
    sim_dir = os.path.join(os.getcwd(), "sim")
    similarities_file = os.path.join(sim_dir, f"similarities_{model_name}.pkl")
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    if  os.path.exists(similarities_file):
        with open(similarities_file, "rb") as f:
            all_similarities = pickle.load(f)
        print("Loaded precomputed similarities and labels from file.")
    else:
        all_similarities = []
        for i in range(length):
            output = data_list[i]["generated_text"].strip().lower()
            sub_sen_list = cut_sentences(output)
            sub_str_list = []
            for sub_sen in sub_sen_list:
                if len(sub_sen) >= 2:
                    sub_str_list.extend(cut_sub_string(sub_sen))
            if len(sub_str_list) != 0:
                    similarities = model.similarity(sub_str_list, uncertain_list, device=device)
            else:
                similarities = [0]
            max_uncertainty = np.max(similarities)
            all_similarities.append(max_uncertainty)
        with open(similarities_file, "wb") as f:
            pickle.dump(all_similarities, f)

    return all_similarities

def eval(data_list,model_name):
    length = len(data_list)
    known_num = 0
    unknown_num = 0
    TP = 0
    FP = 0
    FN = 0
    Acc = 0
    labels=[]
    scores=[]
    if args.evalform == "SimCSE" or args.evalform == "joint_or" or args.evalform == "joint_and":
        scores=get_similarities(data_list,model_name)
    for i in trange(length):
        EigenScore = data_list[i]["EigenScore"]
        NormalizedEntropy = data_list[i]["NormalizedEntropy"]
        known = data_list[i]["known"]
        LexicalSimilarity=data_list[i]["LexicalSimilarity"]
        unknown = known is False
        if known is True:
            known_num += 1
        else:
            unknown_num += 1
        output = data_list[i]["generated_text"].strip().lower()
        pred_unknown = False
        if args.evalform == "SimCSE":
            for uncertain in uncertain_list:
                if uncertain in output:
                    pred_unknown = True

        if pred_unknown is False:
            if args.evalform =="SimCSE":
                if scores[i] > args.threshold_sim:
                    pred_unknown = True
            elif args.evalform =="EigenScore":
                if EigenScore > args.threshold_eigenScore:
                    pred_unknown = True
            elif args.evalform == "NormalizedEntropy":
                if NormalizedEntropy > args.threshold_entropy:
                    pred_unknown = True
            elif args.evalform == "LexicalSimilarity":
                if LexicalSimilarity < args.threshold_LexicalSimilarity:
                    pred_unknown = True
            elif args.evalform == "joint":
                if scores[i] > args.threshold_sim or NormalizedEntropy > args.threshold_entropy:
                    pred_unknown = True

        if unknown is False:
            for ans in data_list[i]["answer"]:
                if ans.strip().lower() in output:
                    Acc += 1
                    pred_unknown = False
                    break
        if unknown is True:
            if pred_unknown is True:
                TP += 1
            else:
                FN += 1
        elif pred_unknown is True:
            FP += 1


    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    Acc=Acc/known_num
    return precision,recall,F1,Acc


if __name__ == "__main__":
    path = args.filename
    filename = os.path.basename(path)
    filename=os.path.splitext(filename)[0]
    uncertain_list = [remove_punctuation(_) for _ in uncertain_list]
    data_list = read_jsonl(path)
    precision, recall, F1, Acc,=eval(data_list,filename)
    print("Filename:", path)
    if args.evalform in threshold_map:
        print(args.evalform, threshold_map[args.evalform])
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", F1)
    print("Acc", Acc )
