import torch
import numpy as np
import sys

def fake_prob(upper, quantile=0.999, mode="exp"):
    if mode == "exp":
        beta = - upper / (np.log(1 - quantile))
        return beta
    elif mode == "lognorm":
        pass
    else:
        sys.exit(f"NOT support mode {mode}")

def fake_data(num_samples=4096, num_t=1, num_d=13, num_s=26, ln_emb=None, text_file=None, quantile=0.999, mode="exp"):
    """
    num_samples: number of samples
    num_t: number of target
    num_d: number of dense features
    num_s: number of sparse features
    ln_emb: embedding sizes
    text_file: output file
    """
    # generate place holder random array, including dense features
    a = np.random.randint(0, 10, (num_t + num_d + num_s, num_samples))
    # generate targets
    a[0, :] = np.random.randint(0, 2, num_samples)
    # generate sparse features
    for k, size in enumerate(ln_emb):
        if size <= 10000:
            # uniqual dist
            a[num_t + num_d + k, :] = np.random.randint(0, size, num_samples)
        else:
            # exp dist
            beta = fake_prob(size, quantile=quantile, mode=mode)
            a[num_t + num_d + k, :] = np.random.exponential(beta, num_samples).astype(np.int)
    a = np.transpose(a)

    # generate print format
    lstr = []
    for _ in range(num_t + num_d):
        lstr.append("%d")
    for _ in range(num_s):
        lstr.append("%x")
    if text_file is not None:
        np.savetxt(text_file, a, fmt=lstr, delimiter='\t',)

def fake_emb(
        sparse_feature_size=None, 
        sparse_feature_num=26, 
        emb_size="", 
        min_cat=2, 
        max_cat=1000000, 
        shuffle=True):
    """
    make up `sparse_feature_num` embedding tables and
    each emb has size (category, `sparse_feature_size`),
    where `min_cat` < category < `max_cat` 
    """
    if sparse_feature_size is None:
        sparse_feature_size = np.random.choice([16,32,64,128,256])
    if emb_size:
        ln_emb = np.fromstring(emb_size, dtype=int, sep="-")
        ln_emb = np.asarray(ln_emb)
        assert ln_emb.size == sparse_feature_num, "sparse_feature_num not match num_emb"
    else:
        emb_list = []
        # hope to generate cat number evenly and reasonable
        p = len(str(max_cat)) # 7
        q = sparse_feature_num // p # 26//7=3
        for i in range(p):
            for _ in range(q):
                s = 10**i if 10**i > min_cat else min_cat
                e = 10**(i+1)-1 if 10**(i+1)-1 < max_cat else max_cat
                # print(s,e)
                size = np.random.randint(s,e) if s<e else s
                emb_list.append(size)
                if len(emb_list) >= sparse_feature_num:
                    break
            if len(emb_list) >= sparse_feature_num:
                break
        while len(emb_list) < sparse_feature_num:
            s = 10**i if 10**i > min_cat else min_cat
            e = 10**(i+1)-1 if 10**(i+1)-1 < max_cat else max_cat
            # print(s,e)
            size = np.random.randint(s,e) if s<e else s
            emb_list.append(size)
        ln_emb = np.array(emb_list)
    if shuffle:
        np.random.shuffle(ln_emb)
    print("=== fake embedding info ===")
    print(f"sparse_feature_size: {sparse_feature_size}")
    print(f"{ln_emb.size} ln_emb: {ln_emb}")
    print("===========================")

    return sparse_feature_size, sparse_feature_num, ln_emb


if __name__=="__main__":
    profile = "terabyte0875"
    num_samples = 4096
    emb_dim=64
    num_dense = 13
    num_sparse = 26
    num_days = 24
    out_dir = "./fake_" + profile + "/"
    out_name = "day_"
    # make emb
    spa_size, spa_num, ln_emb = fake_emb(sparse_feature_size=emb_dim, sparse_feature_num=num_sparse, shuffle=True)
    # make data
    # for k in range(num_days):
        # text_file =  out_dir + out_name + ("" if profile == "kaggle" else str(k))
        # fake_data(num_samples=num_samples, num_d=num_dense, num_s=spa_num, ln_emb=ln_emb, text_file=text_file)
        # print(f"faked data saved at {text_file}")





