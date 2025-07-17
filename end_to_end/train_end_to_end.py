import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from end_to_end.src.dataset import retriever
from src.model.retriever_end import Retriever
from src.dataset.retriever import RetrieverDataset, collate_retriever
from llm_utils import llm_init, llm_inf_all
import torch.nn.functional as F
from src.config.retriever import load_yaml
import re
import string
def load_pretrained_retriever(path, emb_size, device, config_path):
    # YAML 설정 로드
    config = load_yaml(config_path)

    # Retriever 생성 시, config에서 불러온 retriever 설정 적용
    retriever = Retriever(emb_size, **config['retriever'])

    # 모델의 가중치(state_dict) 로드
    checkpoint = torch.load(path, weights_only=False)
    retriever.load_state_dict(checkpoint['model_state_dict'])

    retriever.to(device)
    retriever.train()
    
    return retriever, config 

# Retriever에서 선택된 triple을 추출하는 함수 (자연어 이름으로 명확히 수정)
def get_selected_triplets_from_mask(h_list, r_list, t_list, mask, entity_list, relation_list):
    selected_triplets = [
        (entity_list[h], relation_list[r], entity_list[t]) 
        for h, r, t, selected in zip(h_list, r_list, t_list, mask) if selected == 1
    ]
    return selected_triplets

def construct_prompt(selected_triplets, question):
    triplets_str = "\n".join([f"({h},{r},{t})" for h, r, t in selected_triplets])
    
    prompt = (
        "Based on the triplets from a knowledge graph, please answer the given question.\n\n"
        "Please keep the answers as simple as possible and return all the possible answers "
        'as a list, each with a prefix "ans:".\n\n'
        "Triplets:\n"
        f"{triplets_str}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    
    return prompt

def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

# 논문에서 사용한 정확한 match 함수 재사용
def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1 or s1 in s2  # 양방향 비교로 정확성 높임

def compute_llm_loss(llm_output, ground_truth_answers, logits):
    predicted_answers = [
        line.split("ans:")[-1].strip().lower()
        for line in llm_output[0].split("\n")
        if "ans:" in line.lower()
    ]

    correct = any(
        pred in (gt.lower() for gt in ground_truth_answers)
        for pred in predicted_answers
    )

    target = torch.tensor([1.0 if correct else 0.0]).to(logits.device)

    # logits을 확률로 정확히 변환
    probs = torch.sigmoid(logits)

    # retriever가 선택한 triple 확률 평균값을 기준으로 Loss 계산
    loss = F.binary_cross_entropy(probs.mean().unsqueeze(0), target)

    print(f"Correct match: {correct}")
    print(f"Triple probs mean: {probs.mean().item():.4f}, Target: {target.item()}")
    print(f"LLM Loss: {loss.item():.4f}")

    return loss

def main():
    device = torch.device('cuda:0')

    dataset_config_file = 'configs/retriever/webqsp.yaml'
    dataset_config = load_yaml(dataset_config_file)

    train_set = RetrieverDataset(config=dataset_config, split='train')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)

    emb_size = train_set[0]['q_emb'].shape[-1]
    retriever, retriever_config = load_pretrained_retriever(
        '/home/dongcheon/SubgraphRAG/retrieve/webqsp_Jul12-09:10:28/cpt.pth',
        emb_size,
        device,
        config_path='configs/retriever/webqsp.yaml'
    )
    optimizer = Adam(retriever.parameters(), lr=retriever_config['optimizer']['lr'])

    llm = llm_init(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                   tensor_parallel_size=1,
                   max_seq_len_to_capture=16384,
                   max_tokens=4000,
                   temperature=0)

    epochs = 5
    tau = 1.0  
    
    for epoch in range(epochs):
        total_loss = 0.0
        for sample in tqdm(train_loader):
            (h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
             num_non_text_entities, relation_embs, topic_entity_one_hot,
             target_triple_probs, a_entity_id_list, raw_sample) = [
                item.to(device) if torch.is_tensor(item) else item for item in sample
            ]

            logits, triple_selected_mask = retriever(
                h_id_tensor,
                r_id_tensor,
                t_id_tensor,
                q_emb,
                entity_embs,
                num_non_text_entities,
                relation_embs,
                topic_entity_one_hot,
                tau=tau,
            )

            # ---- 정확히 수정된 부분 (entity/relation 자연어 이름 사용) ----
            entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
            relation_list = raw_sample['relation_list']

            selected_triplets = get_selected_triplets_from_mask(
                h_id_tensor.cpu().numpy(),  # tensor to numpy
                r_id_tensor.cpu().numpy(), 
                t_id_tensor.cpu().numpy(), 
                triple_selected_mask.detach().cpu().numpy(),
                entity_list,
                relation_list
            )
            # ---------------------------------------------------------

            prompt = construct_prompt(selected_triplets, raw_sample['question'])
            prompts = {'user_query': prompt, 'sys_query': 'You are a helpful assistant.'}
            #print("Prompt to LLM:\n", prompts['user_query'])

            llm_output = llm_inf_all(llm, prompts, llm_mode='sys', model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")

            llm_loss = compute_llm_loss(llm_output, raw_sample['a_entity'], logits)

            optimizer.zero_grad()
            llm_loss.backward()
            optimizer.step()

            total_loss += llm_loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

        torch.save({
                'epoch': epoch,
                'config': retriever_config,
                'model_state_dict': retriever.state_dict()
            }, f'end2end_epoch_{epoch}.pth')

        tau = max(0.1, tau * 0.9)

if __name__ == '__main__':
    main()
