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
    experiment_name: str = "exp_bert_qlora_exp2"
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
    num_epochs: int = 5
    learning_rate: float = 5e-5  # Slightly higher LR for LoRA
    weight_decay: float = 0.1 
    use_wandb: bool = True
    verbose: bool = True
    limit_cache: bool = False
    wandb_project: str = "raid-text-detection-exp2"
    wandb_run_name: str = "run1"
    gpus: list = field(default_factory=lambda: [1])
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
    #- Choices: `['homoglyph', 'number', 'article_deletion', 'insert_paragraphs', 'perplexity_misspelling', 'upper_lower', 'whitespace', 'zero_width_space', 'synonym', 'paraphrase', 'alternative_spelling']
    base_human_homoglyphs = full_test.filter(lambda x: x['model']=='human' and x['attack']=='homoglyph')
    human_homoglyphs = base_human_homoglyphs.add_column('label', [0]*len(base_human_homoglyphs))

    base_human_numbers = full_test.filter(lambda x: x['model']=='human' and x['attack']=='number')
    human_numbers = base_human_numbers.add_column('label', [0]*len(base_human_numbers))

    base_human_article_deletion= full_test.filter(lambda x: x['model']=='human' and x['attack']=='article_deletion')
    human_article_deletion = base_human_article_deletion.add_column('label', [0]*len(base_human_article_deletion))
    
    base_human_insert_paragraphs = full_test.filter(lambda x: x['model']=='human' and x['attack']=='insert_paragraphs')
    human_insert_paragraphs = base_human_insert_paragraphs.add_column('label', [0]*len(base_human_insert_paragraphs))

    base_human_perplexity_misspelling = full_test.filter(lambda x: x['model']=='human' and x['attack']=='perplexity_misspelling')
    human_perplexity_misspelling = base_human_perplexity_misspelling.add_column('label', [0]*len(base_human_perplexity_misspelling))
    
    base_human_upper_lower = full_test.filter(lambda x: x['model']=='human' and x['attack']=='upper_lower')
    human_upper_lower = base_human_upper_lower.add_column('label', [0]*len(base_human_upper_lower))

    base_human_whitespace = full_test.filter(lambda x: x['model']=='human' and x['attack']=='whitespace')
    human_whitespace = base_human_whitespace.add_column('label', [0]*len(base_human_whitespace))

    base_human_zero_width_space = full_test.filter(lambda x: x['model']=='human' and x['attack']=='zero_width_space')
    human_zero_width_space = base_human_zero_width_space.add_column('label', [0]*len(base_human_zero_width_space))

    base_human_synonym = full_test.filter(lambda x: x['model']=='human' and x['attack']=='synonym')
    human_synonym = base_human_synonym.add_column('label', [0]*len(base_human_synonym))

    base_human_paraphrase = full_test.filter(lambda x: x['model']=='human' and x['attack']=='paraphrase')
    human_paraphrase = base_human_paraphrase.add_column('label', [0]*len(base_human_paraphrase))

    base_human_alternative_spelling = full_test.filter(lambda x: x['model']=='human' and x['attack']=='alternative_spelling')
    human_alternative_spelling = base_human_alternative_spelling.add_column('label', [0]*len(base_human_alternative_spelling))



    if cfg.verbose:
        print(f"\n📊 Dataset Sizes:")
        print(f"- Train:       {len(train_set)} (human={int(len(human_train))}, ai={int(len(ai_train))})")
        # print(f"- Test clean:  {len(test_clean)} (human={len(h1)}, ai={len(a1)})")
        # print(f"- Test ChatGPT: {len(test_chatgpt)}")
        print(f" Test human_homoglyphs:  {len(human_homoglyphs)}" )
        print(f" Test human_numbers:  {len(human_numbers)}" )
        print(f" Test human_article_deletion:  {len(human_article_deletion)}" )
        print(f" Test human_insert_paragraphs:  {len(human_insert_paragraphs)}" )
        print(f" Test human_perplexity_misspelling:  {len(human_perplexity_misspelling)}" )
        print(f" Test human_upper_lower:  {len(human_upper_lower)}" )
        print(f" Test human_whitespace:  {len(human_whitespace)}" )
        print(f" Test human_zero_width_space:  {len(human_zero_width_space)}" )
        print(f" Test human_synonym:  {len(human_synonym)}" )
        print(f" Test human_paraphrase:  {len(human_paraphrase)}" )
        print(f" Test human_alternative_spelling:  {len(human_alternative_spelling)}" )


        #exit()

    # --- 3) Tokenize all sets ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.tokenizer_cache_path)
    def tokenize_fn(ex):
        return tokenizer(ex['generation'], truncation=True, padding='max_length', max_length=cfg.max_length)

    train_tok = train_set.map(tokenize_fn, batched=True)
    homoglyphs_tok = human_homoglyphs.map(tokenize_fn, batched=True)
    numbers_tok = human_numbers.map(tokenize_fn, batched=True)
    article_deletion_tok = human_article_deletion.map(tokenize_fn, batched=True)
    insert_paragraphs_tok = human_insert_paragraphs.map(tokenize_fn, batched=True)
    perplexity_misspelling_tok = human_perplexity_misspelling.map(tokenize_fn, batched=True)
    upper_lower_tok = human_upper_lower.map(tokenize_fn, batched=True)
    whitespace_tok = human_whitespace.map(tokenize_fn, batched=True)
    zero_width_space_tok = human_zero_width_space.map(tokenize_fn, batched=True)
    synonym_tok = human_synonym.map(tokenize_fn, batched=True)
    paraphrase_tok = human_paraphrase.map(tokenize_fn, batched=True)
    alternative_spelling_tok = human_alternative_spelling.map(tokenize_fn, batched=True)
    

    # Format for PyTorch
    # for ds in [train_tok, clean_tok, chatgpt_tok,gpt3_tok,gpt4_tok,mistral_tok, ll_tok]:
    #     ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
    for ds in [train_tok, homoglyphs_tok, numbers_tok, article_deletion_tok, insert_paragraphs_tok,\
                perplexity_misspelling_tok, upper_lower_tok, whitespace_tok, zero_width_space_tok, synonym_tok,\
                      paraphrase_tok, alternative_spelling_tok]:
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # --- 4) Build DatasetDict ---
    dataset = DatasetDict({
        'train': train_tok,
        'test_homoglyphs': homoglyphs_tok,
        'test_numbers': numbers_tok,
        'test_article_deletion': article_deletion_tok,
        'test_insert_paragraphs': insert_paragraphs_tok,
        'test_perplexity_misspelling': perplexity_misspelling_tok,
        'test_upper_lower': upper_lower_tok,
        'test_whitespace': whitespace_tok,
        'test_zero_width_space': zero_width_space_tok,
        'test_synonym': synonym_tok,
        'test_paraphrase': paraphrase_tok,
        'test_alternative_spelling': alternative_spelling_tok
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

        prefix = ['homoglyphs', 'numbers', 'article_deletion', 'insert_paragraphs', 
                  'perplexity_misspelling', 'upper_lower', 'whitespace', 
                  'zero_width_space', 'synonym', 'paraphrase', 'alternative_spelling'][dataloader_idx]
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
            for k in [
                'test_homoglyphs', 'test_numbers', 'test_article_deletion',
                'test_insert_paragraphs', 'test_perplexity_misspelling',
                'test_upper_lower', 'test_whitespace', 'test_zero_width_space',
                'test_synonym', 'test_paraphrase', 'test_alternative_spelling'
            ] ]

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
