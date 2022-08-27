from transformers import AutoTokenizer, GPT2Tokenizer
import evaluate

def evaluate_metrics(predictions, targets, is_eng=True):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    if is_eng: 
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2").encode
    else: 
        tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian').encode

    bleu_res = bleu.compute(predictions=predictions, references=targets, tokenizer=tokenizer)
    rouge_res = rouge.compute(predictions=predictions, references=targets, tokenizer=tokenizer)

    return {"bleu": bleu_res, "rouge": rouge_res}

if __name__ == "__main__":
    preds = ["سلام دنیا", "این گربه است"]
    target = [['سلام دنیا', 'سلام خوبم'], ['این یک بچه گربه است', 'این یک گربه است']]

    evaluate_metrics(preds, target, is_eng=False)