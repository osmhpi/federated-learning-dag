import ray
from tangle.lab.args import parse_args
from tangle.lab.config import ModelConfiguration, RunConfiguration, LabConfiguration
from tangle.ray.ray_dataset import RayDataset
from .fed_avg import train


def main():
    model_config, run_config, lab_config = parse_args(ModelConfiguration, RunConfiguration, LabConfiguration)

    ray.init(webui_host='0.0.0.0')

    dataset = RayDataset(lab_config, model_config)
    train(dataset, run_config, model_config, lab_config.seed)


if __name__ == "__main__":
    main()
