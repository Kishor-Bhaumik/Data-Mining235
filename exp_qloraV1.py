# raid_text_detector_pipeline.py

import os
import torch
import random
import numpy as np
from dataclasses import dataclass, field
from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, confusion_matrix
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# Config
# ------------------------------
@dataclass
class Config:
    experiment_name: str = "exp_bert_qlora_exp1"
    seed: int = 12
    model_name: str = "bert-base-uncased"
    
    # QLoRA specific parameters
    use_qlora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.2
    target_modules: list = field(default_factory=lambda: ["query", "value"])  # Adjust based on your model
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"

    train_percent_ai: float = 1     
    train_percent_human: float = 1
    test_percent_ai: float = 1
    test_percent_human: float = 1 

    warmup_steps: int = 5  # Add warmup steps
    use_scheduler: bool = True  # Enable learning rate scheduling

    max_length: int = 512
    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 5e-5  # Slightly higher LR for LoRA
    weight_decay: float = 0.1 
    use_wandb: bool = True
    verbose: bool = True
    limit_cache: bool = False
    wandb_project: str = "raid-text-detection"
    wandb_run_name: str = "run1"
    gpus: list = field(default_factory=lambda: [0])
    tokenizer_cache_path: str = "tokenizer"
    tokenized_data_path: str = "tokenized"
    save_dir: str = "./checkpoints"

cfg = Config()

# ------------------------------
# Data Preparation
# ------------------------------
def load_and_prepare_datasets(cfg: Config):
    # Ensure directories exist
    os.makedirs(cfg.tokenizer_cache_path, exist_ok=True)
    os.makedirs(cfg.tokenized_data_path, exist_ok=True)

    # Cache key varies with seed to reshuffle each run if desired
    cache_file = Path(cfg.tokenized_data_path) / f"{cfg.experiment_name}_seed{cfg.seed}.arrow"
    if cache_file.exists() and not cfg.limit_cache:
        if cfg.verbose:
            print("\n✅ Loading cached tokenized dataset...")
        return load_from_disk(str(cache_file))

    if cfg.verbose:
        print("\n🔄 Loading and preparing raw RAID data...")

    # Load full RAID splits
    full_train = load_dataset('liamdugan/raid', split='train')
    full_test  = load_dataset('liamdugan/raid', split='extra')

    # --- 1) Create Training Set: clean human vs clean GPT2 ---
    human_train = full_train.filter(lambda x: x['model']=='human' and x['attack']=='none')
    ai_train    = full_train.filter(lambda x: x['model']=='gpt2' and x['attack']=='none')

    # Sample proportions
    human_train = human_train.shuffle(seed=cfg.seed).select(range(int(len(human_train) * cfg.train_percent_human)))
    ai_train    = ai_train.shuffle(seed=cfg.seed).select(range(int(len(ai_train) * cfg.train_percent_ai)))

    # Label and combine
    human_train = human_train.add_column('label', [0]*len(human_train))
    ai_train    = ai_train.add_column('label', [1]*len(ai_train))
    train_set   = concatenate_datasets([human_train, ai_train]).shuffle(seed=cfg.seed)

    # --- 2) Create four Test Sets ---
    # Base filters
    base_h_clean = full_test.filter(lambda x: x['model']=='human' and x['attack']=='none')
    base_ai_clean = full_test.filter(lambda x: x['model']=='gpt2' and x['attack']=='none')
    # Samples
    h1 = base_h_clean.shuffle(seed=cfg.seed).select(range(int(len(base_h_clean) * cfg.test_percent_human)))
    a1 = base_ai_clean.shuffle(seed=cfg.seed).select(range(int(len(base_ai_clean) * cfg.test_percent_ai)))
    test_clean = concatenate_datasets([h1.add_column('label',[0]*len(h1)), a1.add_column('label',[1]*len(a1))]).shuffle(seed=cfg.seed)

    base_ai_chatgpt= full_test.filter(lambda x: x['model']=='chatgpt' and x['attack']=='none')
    test_chatgpt = base_ai_chatgpt.add_column('label', [1]*len(base_ai_chatgpt))

    base_ai_gpt4= full_test.filter(lambda x: x['model']=='gpt4' and x['attack']=='none')
    test_gpt4 = base_ai_gpt4.add_column('label', [1]*len(base_ai_gpt4))

    base_ai_mistral= full_test.filter(lambda x: x['model']=='mistral' and x['attack']=='none')
    test_mistral = base_ai_mistral.add_column('label', [1]*len(base_ai_mistral))

    base_ai_gpt3= full_test.filter(lambda x: x['model']=='gpt3' and x['attack']=='none')
    test_gpt3 = base_ai_gpt3.add_column('label', [1]*len(base_ai_gpt3))

    base_ai_llama= full_test.filter(lambda x: x['model']=='llama-chat' and x['attack']=='none')
    test_llama = base_ai_llama.add_column('label', [1]*len(base_ai_llama))

    if cfg.verbose:
        print(f"\n📊 Dataset Sizes:")
        print(f"- Train:       {len(train_set)} (human={int(len(human_train))}, ai={int(len(ai_train))})")
        print(f"- Test clean:  {len(test_clean)} (human={len(h1)}, ai={len(a1)})")
        print(f"- Test ChatGPT: {len(test_chatgpt)}")
        print(f"- Test GPT-4:   {len(test_gpt4)}")
        print(f"- Test Mistral: {len(test_mistral)}")
        print(f"- Test GPT-3:   {len(test_gpt3)}")
        print(f"- Test Llama:   {len(test_llama)}")
        #exit()

    # --- 3) Tokenize all sets ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.tokenizer_cache_path)
    def tokenize_fn(ex):
        return tokenizer(ex['generation'], truncation=True, padding='max_length', max_length=cfg.max_length)

    train_tok = train_set.map(tokenize_fn, batched=True)
    clean_tok = test_clean.map(tokenize_fn, batched=True)
    chatgpt_tok = test_chatgpt.map(tokenize_fn, batched=True)
    gpt4_tok = test_gpt4.map(tokenize_fn, batched=True)
    mistral_tok = test_mistral.map(tokenize_fn, batched=True)
    gpt3_tok = test_gpt3.map(tokenize_fn, batched=True)
    ll_tok    = test_llama.map(tokenize_fn, batched=True)

    # Format for PyTorch
    for ds in [train_tok, clean_tok, chatgpt_tok,gpt3_tok,gpt4_tok,mistral_tok, ll_tok]:
        ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])

    # --- 4) Build DatasetDict ---
    dataset = DatasetDict({
        'train': train_tok,
        'test_clean': clean_tok,
        'test_chatgpt': chatgpt_tok,
        'test_gpt4': gpt4_tok,
        'test_mistral': mistral_tok,
        'test_gpt3': gpt3_tok,
        'test_llama': ll_tok
    })
    dataset.save_to_disk(str(cache_file))
    return dataset

# ------------------------------
# Lightning Module
# ------------------------------
class TextClassifier(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        if cfg.use_qlora:
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=cfg.load_in_4bit,
                bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype)
            )
            
            # Load model with quantization
            self.model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model_name, 
                num_labels=2,
                quantization_config=quantization_config,
                device_map="cuda:0"
            )
            
            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.target_modules,
                bias="none",
                use_rslora=True,  # Use RSLoRA for better stability
                init_lora_weights="gaussian"  # More stable initialization
            )
            
            # Apply LoRA to the model
            self.model = get_peft_model(self.model, peft_config)
            
            if cfg.verbose:
                print("🔧 QLoRA Model Info:")
                self.model.print_trainable_parameters()
        else:
            # Standard model loading (your original code)
            self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"]
        )
        labels = batch.pop("label")
        preds = torch.argmax(outputs.logits, dim=1)

        # checi if loss is NaN
        if torch.isnan(outputs.loss):
            raise ValueError("Loss is NaN, check your data and model configuration.")

        cm = confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        if (fp + tn) > 0:
            fpr = fp / (fp + tn)
        else:
            fpr = 0.0
            
        self.log("train_loss", outputs.loss, on_epoch=True)
        self.log("train_fpr", fpr)  # Fixed the space in "train _fpr"
        return outputs.loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch.pop("label")
        outputs = self.model(**batch)
        
        preds = torch.argmax(outputs.logits, dim=1)
        
        acc = accuracy_score(labels.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average='macro')

        cm = confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        if (fp + tn) > 0:
            fpr = fp / (fp + tn)
        else:
            fpr = 0.0

        prefix = ['clean','chatgpt','gpt4','mistral','gpt3','llama'][dataloader_idx]
        self.log(f'test_{prefix}_acc', acc)
        self.log(f'test_{prefix}_precision', precision)
        self.log(f'test_{prefix}_recall', recall)
        self.log(f'test_{prefix}_f1', f1)
        self.log(f'test_{prefix}_fpr', fpr)

    def configure_optimizers(self):
        from transformers import get_linear_schedule_with_warmup
        
        if self.cfg.use_qlora:
            # Only optimize the trainable parameters (LoRA adapters)
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params, 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay,
                eps=1e-8,
                betas=(0.9, 0.999)  # More conservative beta values
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay,
                eps=1e-8,
                betas=(0.9, 0.999)
            )
        if self.cfg.use_scheduler:
            # Calculate total training steps
            total_steps = self.trainer.estimated_stepping_batches
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg.warmup_steps,
                num_training_steps=total_steps
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        
        return optimizer
# ------------------------------
# Data Module
# ------------------------------
class RAIDDataModule(pl.LightningDataModule):
    def __init__(self, dataset, cfg: Config):
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg
        self.collate = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained(cfg.model_name))

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.cfg.batch_size, shuffle=True)

    def test_dataloader(self):
        return [
            DataLoader(self.dataset[k], batch_size=self.cfg.batch_size, shuffle=False)
            for k in ['test_clean','test_chatgpt','test_gpt4','test_mistral','test_gpt3','test_llama']]

# ------------------------------
# Train & Test
# ------------------------------
def train_and_test(cfg: Config):
    dataset = load_and_prepare_datasets(cfg)
    model   = TextClassifier(cfg)
    dm      = RAIDDataModule(dataset, cfg)
    logger  = WandbLogger(project=cfg.wandb_project, name=cfg.experiment_name) if cfg.use_wandb else None

    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator='gpu', devices=cfg.gpus,
        logger=logger,
        log_every_n_steps=10,
        precision=16 if cfg.use_qlora else 32,  # Use mixed precision for QLoRA
        accumulate_grad_batches=2,
        # limit_test_batches=1,
        # limit_train_batches=1,
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)

    # Save artifacts
    save_path = Path(cfg.save_dir) / cfg.experiment_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    if cfg.use_qlora:
        # Save LoRA adapters
        model.model.save_pretrained(save_path)
        print(f"\n✅ QLoRA adapters saved to: {save_path}")
    else:
        # Save full model
        model.model.save_pretrained(save_path)
        
    AutoTokenizer.from_pretrained(cfg.model_name).save_pretrained(save_path)
    print(f"\n✅ Model and tokenizer saved to: {save_path}")

if __name__ == '__main__':
    train_and_test(cfg)
