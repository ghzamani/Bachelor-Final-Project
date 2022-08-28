import torch
from transformers import TrainingArguments
# from clipfamodel import get_model
from transformers import Trainer, trainer_utils
from torch.cuda.amp import autocast
import multiprocessing
import gc
import os
# from clipfadataset import training_set
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np

DATA_FILE = 'dataset.csv'
TEST_SIZE = 0.05
# BATCH_SIZE = 128
IMAGE_SIZE = 224
MAX_LEN = 64  
# MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
# STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

# args = TrainingArguments(
#         "clip-fa",
#         evaluation_strategy="steps",
#         eval_steps=500,
#         logging_steps=500,
#         learning_rate=3e-5,
#         prediction_loss_only=True,
#         weight_decay=0.003,
#         warmup_steps=500,
#         # fp16=True,
#         save_strategy='steps',
#         save_steps=500,
#         save_total_limit=2,
#         load_best_model_at_end=True,
#         # metric_for_best_model='eval_loss',
#         greater_is_better=False,
#         gradient_checkpointing=False,
#         gradient_accumulation_steps=1,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         num_train_epochs=10,
#         report_to='wandb'
#     )

def optimal_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus*4) if num_gpus else num_cpus - 1
    return optimal_value

def clear_gpu():
    torch.clear_autocast_cache()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()

# args.dataloader_num_workers = optimal_workers()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CLIPTrainer(Trainer):
    # # computes loss w/o label smoothing
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     outputs = model(**inputs, return_loss=True)
    #     return outputs["loss"]

    # def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
    #     inputs = self._prepare_inputs(inputs)
    #     with torch.no_grad():
    #         if self.use_amp:
    #             with autocast():
    #                 loss = self.compute_loss(model, inputs)
    #         else:
    #             loss = self.compute_loss(model, inputs)
    #     return (loss, None, None)

    # computes loss w/o label smoothing
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs_copy = {i:inputs[i] for i in inputs if i!='labels'}
        outputs = model(**inputs_copy, return_loss=True)
        # print("\nkeys", outputs.keys())
        # return (outputs['loss'], outputs['logits_per_image'])
        loss = outputs['loss']
        logits = outputs['logits_per_text']
        return (loss, (loss, logits)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        # print("inside prediction step", inputs)
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            # if self.use_amp:
            #     with autocast():
            #         loss = self.compute_loss(model, inputs)
            # else:
            #     loss = self.compute_loss(model, inputs)
            loss, (_, logits) = self.compute_loss(model, inputs, return_outputs=True)
        return (loss, logits, inputs['labels'])

def compute_metrics(p):    
    print("\n***Computing Metrics***")
    pred, labels = p
    pred = np.argmax(pred, axis=1)   
    print(labels.shape)
    print(pred.shape)
    print(labels)
    print(pred)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1} 


def train(model, training_set, epochs, prefix, lr, batch_size):
    args = TrainingArguments(
        prefix,
        # evaluation_strategy="steps",
        evaluation_strategy=trainer_utils.IntervalStrategy.STEPS,
        eval_steps=500,
        logging_steps=500,
        learning_rate=lr,
        # prediction_loss_only=True,
        weight_decay=0.003,
        warmup_steps=500,
        # fp16=True,
        # save_strategy='steps',
        save_strategy=trainer_utils.IntervalStrategy.STEPS,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        # metric_for_best_model='eval_loss',
        greater_is_better=False,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False
    )

    # args.dataloader_num_workers = optimal_workers()

    trainer = CLIPTrainer(model, args,
                          train_dataset=training_set,
                          compute_metrics=compute_metrics,
                          eval_dataset=training_set
                          )
                          
    trainer.train()

    metrics=trainer.evaluate()
    print(metrics)
    model.text_model.save_pretrained(f'{prefix}-{epochs}-text')
    model.vision_model.save_pretrained(f'{prefix}-{epochs}-vision')
    return model



