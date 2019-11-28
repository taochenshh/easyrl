import argparse

from dataclasses import asdict
from dataclasses import fields


def cfg_from_cmd(cfg, parser=None):
    field_types = {field.name: field.type for field in fields(cfg)}
    default_cfg = asdict(cfg)
    if parser is None or not isinstance(parser, argparse.ArgumentParser):
        parser = argparse.ArgumentParser()

    for key, val in default_cfg.items():
        parser.add_argument('--' + key, type=field_types[key], default=val)

    args = parser.parse_args()
    args_dict = vars(args)
    diff_dict = dict()
    for key, val in args_dict.items():
        if val != default_cfg[key]:
            setattr(cfg, key, val)
            diff_dict[key] = val
    if len(diff_dict) > 0:
        setattr(cfg, 'diff_cfg', diff_dict)
    return args
