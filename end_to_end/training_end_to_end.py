import torch
from torch.optim import Adam
import torch.nn.functional as F
from src.model.retriever import Retriever
from src.config.retriever import load_yaml
from prompts import sys_prompt
from src.dataset.retriever import RetrieverDataset
from tqdm import tqdm
from llm_utils import llm_init, llm_inf_all

# --- Utility Functions ---
def gumbel_softmax_select(logits, tau=0.5, hard=False):
    N = logits.shape[0]
    log_probs = torch.stack([logits, torch.zeros_like(logits)], dim=1)
    noise = -torch.log(-torch.log(torch.rand_like(log_probs)))
    gumbel_logits = (log_probs + noise) / tau
    y_soft = F.softmax(gumbel_logits, dim=1)
    if hard:
        idx = y_soft.argmax(dim=1)
        y_hard = torch.nn.functional.one_hot(idx, num_classes=2).float()
        y = (y_hard - y_soft).detach() + y_soft
        mask = y[:, 0]
    else:
        mask = y_soft[:, 0]
    return mask

def extract_ans_lines(llm_output):
    if isinstance(llm_output, list):
        text = llm_output[0]
    else:
        text = llm_output
    lines = text.split('\n')
    answers = [
        line.split('ans:')[-1].strip()
        for line in lines if 'ans:' in line.lower()
    ]
    return [a for a in answers if a]

def normalize(s):
    import string
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join([c for c in s if c not in exclude])
    s = " ".join(s.split())
    return s

def hit(pred_answers, gold_answers):
    pred_norm = [normalize(p) for p in pred_answers]
    gold_norm = [normalize(g) for g in gold_answers]
    for p in pred_norm:
        if any(g == p or g in p or p in g for g in gold_norm):
            return 1
    return 0

def make_prompt(question, selected_triples):
    triple_str = "\n".join(f"({h},{r},{t})" for h, r, t in selected_triples)
    prompt = (
        sys_prompt + "\n\n"
        "Triplets:\n" + triple_str + "\n\n"
        "Question:\n" + question + "\n\n"
        "Answer:\n"
    )
    return prompt

# --- Model and Data Preparation ---
config_file = 'configs/retriever/webqsp.yaml'
ckpt_file = '/home/dongcheon/SubgraphRAG/retrieve/webqsp_Jul12-09:10:28/cpt.pth'
device = torch.device('cuda:0')

config = load_yaml(config_file)
emb_size = config['retriever']['emb_size'] if 'emb_size' in config['retriever'] else 1024

retriever = Retriever(emb_size, **config['retriever']).to(device)
checkpoint = torch.load(ckpt_file, map_location=device)
retriever.load_state_dict(checkpoint['model_state_dict'])
retriever.train()

dataset = RetrieverDataset(config=config, split='train')

llm = llm_init(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    seed=42,
    max_seq_len_to_capture=16384,
    max_tokens=4000,
    temperature=0
)

optimizer = Adam(retriever.parameters(), lr=1e-4)

# --- Joint Training Loop ---
epochs = 5  # 원하는 만큼 반복
for epoch in range(epochs):
    total_hit = 0
    total_samples = 0
    total_loss = 0.0

    for sample in tqdm(dataset, desc=f"Epoch {epoch+1}"):
        h_id_tensor = torch.tensor(sample['h_id_list']).to(device)
        r_id_tensor = torch.tensor(sample['r_id_list']).to(device)
        t_id_tensor = torch.tensor(sample['t_id_list']).to(device)
        q_emb = sample['q_emb'].to(device)
        entity_embs = sample['entity_embs'].to(device)
        num_non_text_entities = len(sample['non_text_entity_list'])
        relation_embs = sample['relation_embs'].to(device)
        topic_entity_one_hot = sample['topic_entity_one_hot'].to(device)

        triple_logits = retriever(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot
        ).squeeze()
        mask = gumbel_softmax_select(triple_logits, tau=0.5, hard=True)
        selected_indices = (mask > 0.1).nonzero(as_tuple=True)[0]
        entity_list = sample['text_entity_list'] + sample['non_text_entity_list']
        relation_list = sample['relation_list']
        selected_triples = [
            (entity_list[sample['h_id_list'][i]],
             relation_list[sample['r_id_list'][i]],
             entity_list[sample['t_id_list'][i]])
            for i in selected_indices
        ]
        if len(selected_triples) == 0:
            continue
        MAX_TRIPLES = 100  # 또는 20, 100 등 실험에 따라 조정
        if len(selected_triples) > MAX_TRIPLES:
            selected_triples = selected_triples[:MAX_TRIPLES]
        prompt = make_prompt(sample['question'], selected_triples)
        gold_answers = sample['a_entity']

        with torch.no_grad():
            prompts = {'user_query': prompt, 'sys_query': sys_prompt}
            llm_output = llm_inf_all(
                llm, prompts, llm_mode='sys', model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
            )
        pred_answers = extract_ans_lines(llm_output)
        this_hit = hit(pred_answers, gold_answers)
        total_hit += this_hit
        triple_labels = torch.tensor(sample['target_triple_probs'], device=triple_logits.device).float()
        bce_loss = F.binary_cross_entropy_with_logits(triple_logits, triple_labels)


        probs = torch.sigmoid(triple_logits)
        pred_prob = probs.mean().unsqueeze(0)
        target = torch.tensor([float(this_hit)], device=pred_prob.device)
        reward_loss = F.binary_cross_entropy(pred_prob, target)

        rho = 0.7  # 실험적으로 0.5~0.9 등으로 변경 가능
        loss = rho * bce_loss + (1 - rho) * reward_loss
        optimizer.zero_grad()

# retriever의 첫 번째 파라미터만 샘플로 체크(모든 파라미터 반복도 가능)
        first_param_name, first_param = next(retriever.named_parameters())
        old_param = first_param.clone().detach()

        loss.backward()
        optimizer.step()

        param_change = (old_param - first_param).abs().max().item()
        print(f"[{first_param_name}] param max change after update: {param_change:.8f}")
        print((old_param - first_param).abs().max())


        total_loss += loss.item()
        total_samples += 1
    save_path = f"hard_joint_ranking_epoch{epoch+1}.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': retriever.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, save_path)
    print(f"[INFO] Retriever model saved to {save_path}")
    print(f"Epoch {epoch+1} done | Avg Loss: {total_loss/total_samples:.4f}, Hit accuracy: {total_hit/total_samples:.3f}")