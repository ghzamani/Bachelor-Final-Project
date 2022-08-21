import torch
from transformers import TrainingArguments
from clipfamodel import get_model
from transformers import Trainer
from torch.cuda.amp import autocast
import multiprocessing
import gc
import os
from clipfadataset import training_set

DATA_FILE = 'dataset.csv'
TEST_SIZE = 0.05
BATCH_SIZE = 128
IMAGE_SIZE = 224
MAX_LEN = 64  
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

args = TrainingArguments(
        "clip-fa",
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        learning_rate=3e-5,
        prediction_loss_only=True,
        weight_decay=0.003,
        warmup_steps=500,
        # fp16=True,
        save_strategy='steps',
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        # metric_for_best_model='eval_loss',
        greater_is_better=False,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=10,
        report_to='wandb'
    )

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
    # computes loss w/o label smoothing
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs, return_loss=True)
        return outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)
        return (loss, None, None)
clip = get_model()
trainer = CLIPTrainer(clip, args,
                          train_dataset=training_set,
                        #   eval_dataset=test_ds,
                          )
                          
trainer.train()

clip.text_model.save_pretrained('clip-fa-text')
clip.vision_model.save_pretrained('clip-fa-vision')


