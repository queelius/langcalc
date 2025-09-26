from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False) # the tokenizer should match that of the index you load below
engine = InfiniGramEngine(index_dir='index/v4_pileval_llama', eos_token_id=tokenizer.eos_token_id) # please replace index_dir with the local directory where you store the index

input_ids = tokenizer.encode('natural language processing')
print(input_ids)
#[5613, 4086, 9068]

out = engine.count(input_ids=input_ids)

print(out)
# {'count': 76, 'approx': False}


print(engine.count(input_ids=[]))
#{'count': 393769120, 'approx': False}



# natural language processing OR artificial intelligence
cnf = [
    [tokenizer.encode('natural language processing'), tokenizer.encode('artificial intelligence')]
]
print(cnf)
# [[[5613, 4086, 9068], [23116, 21082]]]

print(engine.count_cnf(cnf=cnf))
# {'count': 499, 'approx': False}




print("-"*50)

input_ids = tokenizer.encode('natural language processing')
print(input_ids)
# [5613, 4086, 9068]

print(engine.prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1]))
#{'prompt_cnt': 257, 'cont_cnt': 76, 'prob': 0.29571984435797666}









# (natural language processing OR artificial intelligence) AND deep learning
cnf = [
   [tokenizer.encode('natural language processing'), tokenizer.encode('artificial intelligence')],
   [tokenizer.encode('deep learning')]
]
print(cnf)
# [[[5613, 4086, 9068], [23116, 21082]], [[6483, 6509]]]

print(engine.count_cnf(cnf=cnf))
#{'count': 19, 'approx': False}
