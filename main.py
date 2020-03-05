from train import train
from create_index import create_index
from online_inference import online_inference


if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	cfg_load = OmegaConf.load(f'config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	model_folder = cfg.dir

	os.makedirs(model_folder, exist_ok=True)

	# save config
	OmegaConf.save(cfg, f'{model_folder}/config.yaml')
	# set model_folder
	cfg.model_folder = model_folder
	train(cfg)
