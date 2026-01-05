import pandas as pd

def read_nonbond_params(itp_path):
    with open(itp_path, 'r') as f:
        lines = f.readlines()

    # 找到 [ nonbond_params ] 的起始行
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('[ nonbond_params ]'):
            start = i
            break
    if start is None:
        raise ValueError('未找到 [ nonbond_params ] 部分')

    # 读取参数行
    params = []
    for line in lines[start+1:]:
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('['):
            if line.startswith('['):  # 下一个section开始
                break
            continue
        # 只保留前6列（i, j, funda, c6, c12, 注释可选）
        parts = line.split(';')[0].split()
        if len(parts) >= 5:
            params.append(parts[:5])

    df = pd.DataFrame(params, columns=['i', 'j', 'funda', 's', 'eps'])
    return df

# 用法
itp_file = '/lustre/grp/gyqlab/lism/cryostar/martini_test/na-tutorials/rna-tutorial/dsRNA-setup/martini_v2.1-dna.itp'
df_dna = read_nonbond_params(itp_file)