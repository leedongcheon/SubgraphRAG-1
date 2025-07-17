import torch
from torch.optim import Adam
import torch.nn.functional as F
from src.model.retriever import Retriever
from src.config.retriever import load_yaml
from src.dataset.retriever import RetrieverDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import string
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gumbel_softmax_select(logits, tau=0.5, hard=True):
    log_probs = torch.stack([logits, torch.zeros_like(logits)], dim=1)
    noise = -torch.log(-torch.log(torch.rand_like(log_probs)))
    gumbel_logits = (log_probs + noise) / tau
    y_soft = F.softmax(gumbel_logits, dim=1)
    if hard:
        idx = y_soft.argmax(dim=1)
        y_hard = F.one_hot(idx, num_classes=2).float()
        y = (y_hard - y_soft).detach() + y_soft
        mask = y[:, 0]
    else:
        mask = y_soft[:, 0]
    return mask

config = load_yaml('configs/retriever/webqsp.yaml')
emb_size = config['retriever'].get('emb_size', 1024)
retriever = Retriever(emb_size, **config['retriever']).to(device)
retriever.load_state_dict(torch.load('/home/dongcheon/SubgraphRAG/retrieve/webqsp_Jul12-09:10:28/cpt.pth')['model_state_dict'])
retriever.train()

llm_ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_ckpt, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm_model = AutoModelForCausalLM.from_pretrained(
    llm_ckpt, torch_dtype=torch.bfloat16, device_map="auto"
)
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)
llm_model = get_peft_model(llm_model, lora_config).to(device).train()

for name, param in llm_model.named_parameters():
    param.requires_grad = "lora_" in name

llm_embedding_dim = llm_model.config.hidden_size
semantic_proj = torch.nn.Linear(emb_size, llm_embedding_dim).to(device)
triple_proj = torch.nn.Linear(4 * emb_size, llm_embedding_dim).to(device)
final_proj = torch.nn.Linear(llm_embedding_dim * 2, llm_embedding_dim).to(device)

params = (
    list(retriever.parameters()) +
    list(semantic_proj.parameters()) +
    list(triple_proj.parameters()) +
    list(final_proj.parameters()) +
    [p for n, p in llm_model.named_parameters() if p.requires_grad]
)
optimizer = Adam(params, lr=1e-4)

epochs = 10
rho = 0.7

dataset = RetrieverDataset(config=config, split='train')
val_dataset = RetrieverDataset(config=config, split='val')
for epoch in range(epochs):
    total_loss = 0.0
    for sample in tqdm(dataset, desc=f"Epoch {epoch+1}"):
        h_id_tensor = torch.tensor(sample['h_id_list']).to(device)
        r_id_tensor = torch.tensor(sample['r_id_list']).to(device)
        t_id_tensor = torch.tensor(sample['t_id_list']).to(device)

        triple_logits, triple_embs = retriever(
            h_id_tensor, r_id_tensor, t_id_tensor, sample['q_emb'].to(device),
            sample['entity_embs'].to(device), len(sample['non_text_entity_list']),
            sample['relation_embs'].to(device), sample['topic_entity_one_hot'].to(device),
            return_triple_emb=True,
            return_semantic_only=True
        )

        mask = gumbel_softmax_select(triple_logits, tau=0.5, hard=True) 
        num_selected = mask.sum().item()
        print(f"Selected Triples 개수: {num_selected}")
        # 디버깅: 차원 출력
        #print(f"triple_embs.shape: {triple_embs.shape}, mask.shape: {mask.shape}")
        
        # Mask 적용
        selected_triple_emb = (triple_embs * mask).sum(dim=0)

        semantic_emb = sample['q_emb'].to(device)
        # 수정된 semantic embedding projection
        semantic_emb_proj = semantic_proj(semantic_emb)
        if semantic_emb_proj.dim() == 1:
            semantic_emb_proj = semantic_emb_proj.unsqueeze(0)  # 정확히 2차원 (1, emb_dim)으로 맞춤

        # 수정된 triple embedding projection
        triple_emb_proj = triple_proj(selected_triple_emb)
        if triple_emb_proj.dim() == 1:
            triple_emb_proj = triple_emb_proj.unsqueeze(0)

        # 디버깅 출력 (반드시 이 형태로 나와야 함!)
        # print("semantic_emb_proj.shape:", semantic_emb_proj.shape)  # (1, emb_dim)
        # print("triple_emb_proj.shape:", triple_emb_proj.shape)      # (1, emb_dim)

        # 최종 embedding concat
        final_input_emb = torch.cat([semantic_emb_proj, triple_emb_proj], dim=-1)  # (1, 2*emb_dim)
        final_input_emb = final_proj(final_input_emb).unsqueeze(1)                 # (1, 1, emb_dim)
        final_input_emb = final_input_emb.to(torch.bfloat16)

        labels_text = "\\n".join([f"ans: {ans}" for ans in sample['a_entity']])
        labels_ids = tokenizer(labels_text, return_tensors='pt').input_ids.to(device)

        # 디버깅: labels 길이 출력
        # print("labels_ids.shape:", labels_ids.shape)

        # 입력 embedding을 labels와 동일한 길이로 반복
        final_input_emb = final_input_emb.repeat(1, labels_ids.shape[1], 1)  # (1, labels_seq_len, emb_dim)

        # labels 설정 (정확히 동일한 길이로)
        labels = labels_ids.clone()

        # 디버깅: 최종 shape 출력
        # print("final_input_emb.shape:", final_input_emb.shape)
        # print("labels.shape:", labels.shape)

        # LLM forward
        llm_out = llm_model(inputs_embeds=final_input_emb, labels=labels)
        llm_loss = llm_out.loss
        target_triple_probs = sample['target_triple_probs']
        if torch.is_tensor(target_triple_probs):
            target_triple_probs = target_triple_probs.clone().detach().float().to(mask.device)
        else:
            target_triple_probs = torch.tensor(target_triple_probs, device=mask.device).float()

        target_triple_probs = target_triple_probs.view(-1, 1)  # 차원 맞추기

        # Retriever 손실
        bce_loss = F.binary_cross_entropy(mask, target_triple_probs)


        retriever_loss = rho * bce_loss + (1 - rho) * llm_loss

        optimizer.zero_grad()
        retriever_loss.backward()
        optimizer.step()

        total_loss += retriever_loss.item()

    print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(dataset)}")

    torch.save({
        'epoch': epoch + 1,
        'retriever_state_dict': retriever.state_dict(),
        'llm_lora_state_dict': llm_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, f"joint_lora_epoch{epoch+1}.pth")
