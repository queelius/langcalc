from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

# Load the GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2", add_bos_token=False, add_eos_token=False)

# Initialize the Infini-gram engine with the GPT-2 index
engine = InfiniGramEngine(index_dir='./v4_pileval_gpt2', eos_token_id=tokenizer.eos_token_id)


input_ids = tokenizer.encode('natural language processing')
result = engine.count(input_ids=input_ids)
print(result)


cnf = [
    [tokenizer.encode('natural language processing')],
    [tokenizer.encode('deep learning')],
]
result = engine.count_cnf(cnf=cnf)
print(result)




input_ids = tokenizer.encode('natural language')
result = engine.ntd(prompt_ids=input_ids)
print(result)




input_ids = tokenizer.encode('natural language processing')
result = engine.search_docs(input_ids=input_ids, maxnum=1, max_disp_len=10)
print(result)
