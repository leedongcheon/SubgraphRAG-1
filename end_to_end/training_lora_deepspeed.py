import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from src.model.retriever import Retriever
from src.config.retriever import load_yaml
from src.dataset.retriever import RetrieverDataset

# ----- Joint Model -----
class JointModel(nn.Module):
    def __init__(self, retriever, llm_model):
        super().__init__()
        self.retriever = retriever
        self.llm_model = llm_model

    def forward(self, 
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, num_non_text_entity_list,
                relation_embs, topic_entity_one_hot, target_triple_probs,
                input_ids, attention_mask, labels):
        # Retriever forward
        triple_logits = self.retriever(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, num_non_text_entity_list,
            relation_embs, topic_entity_one_hot
        ).squeeze()
        # --- BCE Loss로 변경!
        retriever_loss = F.binary_cross_entropy_with_logits(triple_logits, target_triple_probs)

        # LLM + LoRA loss
        outputs = self.llm_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        llm_loss = outputs.loss

        loss = retriever_loss + llm_loss
        return {"loss": loss, "retriever_loss": retriever_loss, "llm_loss": llm_loss}

# ----- 데이터 Collate (retriever+LLM input 모두) -----
def joint_collate_fn(batch, tokenizer):
    # retriever 인풋
    h_id_tensor = torch.stack([torch.tensor(b['h_id_list']) for b in batch])
    r_id_tensor = torch.stack([torch.tensor(b['r_id_list']) for b in batch])
    t_id_tensor = torch.stack([torch.tensor(b['t_id_list']) for b in batch])
    q_emb = torch.stack([torch.tensor(b['q_emb']) for b in batch])
    entity_embs = torch.stack([torch.tensor(b['entity_embs']) for b in batch])
    num_non_text_entity_list = torch.tensor([len(b['non_text_entity_list']) for b in batch])
    relation_embs = torch.stack([torch.tensor(b['relation_embs']) for b in batch])
    topic_entity_one_hot = torch.stack([torch.tensor(b['topic_entity_one_hot']) for b in batch])
    target_triple_probs = torch.stack([torch.tensor(b['target_triple_probs'], dtype=torch.float32) for b in batch])

    # LLM input/label
    prompts = [b['prompt'] for b in batch]
    labels_text = [b['labels_text'] for b in batch]

    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    labels = tokenizer(labels_text, return_tensors='pt', padding=True, truncation=True, max_length=64).input_ids
    # prompt 길이 masking
    for i in range(labels.size(0)):
        prompt_len = inputs['input_ids'][i].ne(tokenizer.pad_token_id).sum().item()
        labels[i, :prompt_len] = -100

    return {
        "h_id_tensor": h_id_tensor,
        "r_id_tensor": r_id_tensor,
        "t_id_tensor": t_id_tensor,
        "q_emb": q_emb,
        "entity_embs": entity_embs,
        "num_non_text_entity_list": num_non_text_entity_list,
        "relation_embs": relation_embs,
        "topic_entity_one_hot": topic_entity_one_hot,
        "target_triple_probs": target_triple_probs,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }

# ----- 모델/토크나이저/데이터셋 -----
config = load_yaml('configs/retriever/webqsp.yaml')
emb_size = config['retriever'].get('emb_size', 1024)
retriever = Retriever(emb_size, **config['retriever'])
retriever.load_state_dict(torch.load('/home/dongcheon/SubgraphRAG/retrieve/webqsp_Jul12-09:10:28/cpt.pth')['model_state_dict'])
retriever.train()

llm_ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_ckpt, use_fast=True)
llm_model = AutoModelForCausalLM.from_pretrained(llm_ckpt, torch_dtype=torch.bfloat16, device_map="auto")
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)
llm_model = get_peft_model(llm_model, lora_config).train()
joint_model = JointModel(retriever, llm_model)

# ----- RetrieverDataset 전처리 → prompt/labels_text 생성
def preprocess(sample):
    prompt = make_prompt(sample['question'], sample['text_entity_list'][:3])  # or triple 구조
    labels_text = "\n".join([f"ans: {ans}" for ans in sample['a_entity']])
    sample['prompt'] = prompt
    sample['labels_text'] = labels_text
    return sample

train_samples = [preprocess(s) for s in dataset]      # dataset은 RetrieverDataset
val_samples = [preprocess(s) for s in val_dataset]

from datasets import Dataset
train_dataset = Dataset.from_list(train_samples)
val_dataset = Dataset.from_list(val_samples)

# ----- Trainer/TrainingArguments -----
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    fp16=True,
    deepspeed="ds_config.json",
    save_steps=100,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=joint_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=lambda batch: joint_collate_fn(batch, tokenizer)
)

trainer.train()
