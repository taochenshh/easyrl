import argparse

from dataclasses import asdict
from dataclasses import fields


def cfg_from_cmd(cfg):
    field_types = {field.name: field.type for field in fields(cfg)}
    default_cfg = asdict(cfg)
    parser = argparse.ArgumentParser()

    for key, val in default_cfg.items():
        parser.add_argument('--' + key, type=field_types[key], default=val)

    args = parser.parse_args()
    args_dict = vars(args)
    for key, val in args_dict.items():
        setattr(cfg, key, val)
