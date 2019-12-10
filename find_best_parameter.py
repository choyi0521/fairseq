from bayes_opt import BayesianOptimization
import subprocess

execute1 = "python3 train.py data-bin/iwslt14.tokenized.de-en --encoder-conv-type ddynamic --decoder-conv-type ddynamic --optimizer adam --lr 0.0005 --source-lang de --target-lang en --max-tokens 4000 --no-progress-bar --log-interval 100 --min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler inverse_sqrt --ddp-backend=no_c10d --max-update 50000 --warmup-updates 4000 --warmup-init-lr 1e-07 --keep-last-epochs 10 -a lightconv_iwslt_de_en      --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1     --encoder-glu 0 --decoder-glu 0     --encoder-attention-heads"
execute2 = "  --decoder-attention-heads"
execute3 = "  --encoder-attention-query-heads"
execute4 = "  --decoder-attention-query-heads"
execute5 = "  --encoder-query-dim"
execute6 = "  --decoder-query-dim"
execute7 = "  --encoder-layers 7 --encoder-kernel-size-list [3,7,15,31,31,31,31]"
# Let's start by definying our function, bounds, and instanciating an optimization object.
pbounds = {'attention_heads': (1,3), 	'attention_query_heads': (1,3), 'query_dim':(6,9)}

def black_box_function(attention_heads, attention_query_heads, query_dim):
	attention_heads = 2 ** int(attention_heads)
	attention_query_heads = 2 ** int(attention_query_heads)
	query_dim = 2 ** int(query_dim)
	execute = execute1 + " " + str(attention_heads) + execute2 + " " + str(attention_heads) + execute3 + " " + str(attention_query_heads) + execute4 + " " + str(attention_query_heads) + execute5 + " " + str(query_dim) + execute6 + " " + str(query_dim) + execute7
	output = subprocess.check_output(execute.split())
	output = str(output).split("valid on 'valid' subset | loss ",3)[3][:6]
	# n epoch 
	# output = str(output).split("valid on 'valid' subset | loss ",n)[n][:6]
	print(output)
	return - float(output)

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)


optimizer.maximize(
    init_points=2,
    n_iter=3,
)

print(optimizer.max)
