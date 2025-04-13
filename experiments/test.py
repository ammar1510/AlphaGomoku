import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../cfg", config_name="pong_ppo_config") # Corrected path
def main(cfg: DictConfig):
    print("type of cfg: ", type(cfg))
    cfg_obj = OmegaConf.to_object(cfg)
    print("type of cfg_obj: ", type(cfg_obj))
    print("cfg_obj: ", cfg_obj)

if __name__ == "__main__":
    main()