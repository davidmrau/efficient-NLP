
import numpy as np
from enlp.file_interface import FileInterface


fname = 'data/robust04/robust04_AOL_anserini_top_1000_qld_no_stem'
n = 2000



f = FileInterface(fname)
ran = np.random.choice(range(len(f)), size=2000, replace=False)

with open(f'{fname}_{n}', 'w') as fou:
	for i in ran:
		res = f.get_query_lines(i)
		for l in res:
			fou.write(l)
