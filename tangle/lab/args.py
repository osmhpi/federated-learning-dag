import argparse

def parse_args(*config_cls):
    parser = argparse.ArgumentParser()

    configs = [c() for c in config_cls]

    for c in configs:
        c.define_args(parser)

    result = parser.parse_args()

    for c in configs:
        c.parse(result)

    return configs

