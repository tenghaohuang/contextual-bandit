import json
import torch
import nltk
import argparse
import logging
from IPython import embed
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.nn import CrossEntropyLoss

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def process_story(story):
    sentences = nltk.sent_tokenize(story)
    context_sent = [("", sentences[0])]
    context_sent += [(" ".join(sentences[:i]), sentences[i]) for i in range(1, len(sentences))]

    return context_sent

def mask_context(token_ids, mask_len):
    mask = torch.zeros_like(token_ids)
    mask[(torch.arange(token_ids.shape[0]), mask_len)] = 1
    mask = 1 - mask.cumsum(dim=-1)
    token_ids[mask.bool()] = -100

    return token_ids

def compute_ppl(input_ids, target_ids, model):
    with torch.no_grad():
        no_sents, seq_lens = input_ids.shape
        shift_labels = target_ids[...,1:].contiguous().view(-1)
        lm_logits = model(input_ids).logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(no_sents,-1)
        loss = loss.sum(dim=1) / (loss != 0.0).sum(dim=1)

    perplexity = torch.exp(loss)
    return perplexity
 
def get_story_ppl(context_sentence, tokenizer, model, device):

    contexts = [item[0] for item in context_sentence]
    # sentences = [item[1] for item in context_sentence]
    input_ids = tokenizer(context_sentence, return_tensors="pt", padding=True)["input_ids"].to(device)

    context_ids = tokenizer(contexts, return_tensors="pt", padding=True)["input_ids"].to(device)
    context_lens = torch.sum(context_ids != tokenizer.pad_token_id, dim=1)

    target_ids = input_ids.clone()
    # set pad token labels to -100 (avoid computing loss on these tokens)
    target_ids[target_ids == tokenizer.pad_token_id] = -100
    target_ids = mask_context(target_ids, context_lens)

    sentences_ppl = compute_ppl(input_ids, target_ids, model)

    return sentences_ppl



def init_model(model_name: str,
               device: torch.device):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    logger.info(f'Initializing {model_name}')
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = tokenizer.unk_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    # Added for the CUDA error: CUBLAS_STATUS_NOT_INITIALIZED error when using a finetuned gpt model.
    model.resize_token_embeddings(len(tokenizer))
    # !
    model.to(device)
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt2-medium", type=str, required=False, help="language model to use")
    # parser.add_argument("--dataset_file", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_dir", default=None, type=str, required=False, help="Out directory to save ppl scores")
    parser.add_argument("--device", default=-1, type=int, required=True, help="GPU device")

    args = parser.parse_args()
    logger.info(args)

    # Load the language model
    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    model, tokenizer = init_model(args.model_name, device)
    context_sentences = process_story("I have an apple. It is exciting.")
    ppl_scores = get_story_ppl(context_sentences, tokenizer, model, device)
    print(ppl_scores)


    story_text = "Tom was eating hotdog on a train. He"
    story_text2 = "Tom was eating hotdog on a train. d"
    story_text3 = "Tom was eating hotdog on a train. B"
    context_sentences =  process_story(story_text)
    ppl_scores = get_story_ppl(context_sentences, tokenizer, model, device)

    print(ppl_scores)
    context_sentences =  process_story(story_text2)
    ppl_scores = get_story_ppl(context_sentences, tokenizer, model, device)

    print(ppl_scores)
    context_sentences =  process_story(story_text3)
    ppl_scores = get_story_ppl(context_sentences, tokenizer, model, device)
    #
    # print(ppl_scores)
    # import pickle
    # file = open("storySpace_only3.pickle",'rb')
    # stories = pickle.load(file)
    # record = []
    # for story in stories:
    #     story = story + "."
    #     context_sentences =  process_story(story)
    #     ppl_scores = get_story_ppl(context_sentences, tokenizer, model, device)
    #     record.append((story,ppl_scores[1]))
    # record.sort(key=lambda tup: tup[1],reverse=True)
    # handle = open("stories_ppls.pickle","wb")
    # pickle.dump(record, handle)




if __name__ == '__main__':
    main()