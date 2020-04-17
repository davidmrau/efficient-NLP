import sys
from omegaconf import OmegaConf
import os




def special(cfg):
    special = ''
    if 'large_out_bias' in cfg:
        special += '+bias' + ' '
    if 'balance_scalar' in cfg:
        if float(cfg.balance_scalar) > 0:
            special += f'bal{cfg.balance_scalar}' + ' '
    if cfg.model == 'tf':
        special += f'{cfg.tf.pooling_method}' + ' '
    if 'act_func' in cfg:
        special += f'{cfg.act_func}' + ' '

    return special + '\t'

def tf2str(cfg):
    res = ''
    res += f'{cfg.model}\t'
    res += f'{cfg.tf.num_attention_heads}\t'
    # special
    res += special(cfg)

    res += f'{cfg.embedding}\t'
    res += f'{cfg.batch_size}\t'
    res += f'{cfg.sparse_dimensions}\t'

    return res


def snrm2str(cfg):
    res = ''
    res += f'{cfg.model}\t'
    res += f'{cfg.snrm.hidden_sizes}\t'
    # special
    #res += special(cfg)
    res += f'{cfg.embedding}\t'
    res += f'{cfg.batch_size}\t'
    res += f'{cfg.sparse_dimensions}\t'
    return res

def ranking_results2str(ranking_result_lines):
    res = ''
    if ranking_result_lines == '':
        return res 
    for line in ranking_result_lines[:-1]:
        line = line.strip()
        res += line.split('\t')[1].strip() + '\t'

    res += ranking_result_lines[-1].split('\t')[1][12:19] + '\t'
    return res


experiments_folder = sys.argv[1]

outfile = f'{experiments_folder}/collected_results.txt'
ranking_results = 'ranking_results.txt'
subdirs = [os.path.join(experiments_folder, o) for o in os.listdir(experiments_folder) if os.path.isdir(os.path.join(experiments_folder,o))]


with open(outfile, 'w') as f:
    for model_path in subdirs:
        res = ''
        try:
            ranking_results_fname = f'{model_path}/{ranking_results}'
            ranking_results_str = open(ranking_results_fname, 'r').readlines()

            cfg = OmegaConf.load(f'{model_path}/config.yaml')
            if cfg.model == 'snrm':
                res = snrm2str(cfg)
            elif cfg.model == 'tf':
                res = tf2str(cfg)
            else:
                raise NotImplementedError()
            res += ranking_results2str(ranking_results_str)
            print(res)
            f.write(res + '\n')

        except Exception as e:
            print(e)
            print(f'No results file found, skipping {ranking_results_fname}')
            continue


