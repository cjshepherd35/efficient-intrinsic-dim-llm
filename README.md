This model uses shared matrices and updates a mutually exclusive subset of each shared matrix with the gradients from each layer. meaning for one shared matrix there is a subset of the parameters that gets updated with the gradients from the first layer, a second subset of parameters of the matrix gets updated with gradients from the second layer, and so on. based on the idea by Uber AI about intrinsic dimension, that the data manifold lies in a smaller dimension than the full parameterization of the weight matrix. 

based on Nanogpt from Karpathy, uses same input.txt file as the other llm stuff i have. It is shakespeare dataset. The file efficientintrinsdimattention.py only reuses key, query, value, and outptu projections while efficientintrdim_ffwd.py also reuses the feedforward class linear layers. 


****updated
added files using wiki-103 dataset. timedwikikarpath.py is the standard transformer and I compare its losses with about 130 million parameters vs the newest test of weight reuse called eff_intr_dim_engrammlp_wiki.py, for efficient intrinsic dimension engram mlp. this model uses about 40 million parameters currently with almost the same losses at less than half the time. I should be able to decrease  the parameter count by using rmsnorm rather than a layernorm layer. 

eff_intr_dim_engrammlp_wiki.py
size of model 127638304 with 3e-4 lr
step 0: train loss 6.8625, val loss 6.8738
step 5000: train loss 3.0917, val loss 3.1901
step 10000: train loss 2.8175, val loss 3.0213
step 15000: train loss 2.6865, val loss 2.9243
step 20000: train loss 2.6020, val loss 2.8335
Training time: 3735.05 seconds

eff_intr_dim_engrammlp_wiki.py with lr 1e-3
size of model (intrinsic dimensions & engram adapted): 41581344
step 0: train loss 6.8437, val loss 6.8366 (train time 0.65s)
step 5000: train loss 3.0087, val loss 3.2218 (train time 360.73s)
step 10000: train loss 2.8119, val loss 3.0808 (train time 723.23s)
step 15000: train loss 2.7321, val loss 3.0028 (train time 1085.92s)
step 20000: train loss 2.6782, val loss 3.0176 (train time 1449.85s)
Total training time: 1449.88 seconds



****





regular karpathy bpe transformer
size of model 1294480
step 0: train loss 6.1343, val loss 6.1258
step 1000: train loss 3.1633, val loss 3.3627
step 2000: train loss 2.8097, val loss 3.1427
step 3000: train loss 2.6636, val loss 3.0155
step 4000: train loss 2.5857, val loss 2.9458
step 5000: train loss 2.5318, val loss 2.8910

efficientintrindimattention.py
size of model (intrinsic dimensions adapted): 1031568
step 0: train loss 6.1556, val loss 6.1594
step 1000: train loss 3.2105, val loss 3.4197
step 2000: train loss 2.8856, val loss 3.1790
step 3000: train loss 2.7437, val loss 3.0554
step 4000: train loss 2.6348, val loss 3.0024
step 5000: train loss 2.5772, val loss 2.9388

efficientintrdim_incl_ffwd.py
size of model (intrinsic dimensions adapted): 503440
step 0: train loss 6.1359, val loss 6.1461
step 1000: train loss 3.2813, val loss 3.4563
step 2000: train loss 2.9590, val loss 3.2794
step 3000: train loss 2.8345, val loss 3.1523
step 4000: train loss 2.7558, val loss 3.0846
step 5000: train loss 2.6956, val loss 3.0097



added newest version, heff_moe.py which with 40 million params is pretty equivalent to a comparable standard model with 400 mill params. 
size of model 44370056
size of model if weights were not reused across layers 223643784
step 0: train loss 7.0990, val loss 7.0860
step 5000: train loss 2.6543, val loss 3.0712
step 10000: train loss 2.3880, val loss 3.0423
step 15000: train loss 2.2444, val loss 3.0161
step 20000: train loss 2.1480, val loss 2.9867
Training time: 9629.79 seconds

versus similar standard, multiheaded latent attention model with  mixture of experts called wikiv2frontier.py 
size of model 433389704
step 0: train loss 7.0819, val loss 7.0794
step 5000: train loss 2.5302, val loss 3.0038
step 10000: train loss 2.0188, val loss 3.0090
step 15000: train loss 1.6367, val loss 3.1032
