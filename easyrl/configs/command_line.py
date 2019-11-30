import argparse

from dataclasses import asdict
from dataclasses import fields


def cfg_from_cmd(cfg, parser=None):
    field_types = {field.name: field.type for field in fields(cfg)}
    default_cfg = asdict(cfg)
    if parser is None or not isinstance(parser, argparse.ArgumentParser):
        parser = argparse.ArgumentParser()

    for key, val in default_cfg.items():
        # add try except here so that we can define the
        # configs in the parser manually with
        # different default values
        try:
            parser.add_argument('--' + key, type=field_types[key], default=val)
        except argparse.ArgumentError:
            pass

    args = parser.parse_args()
    args_dict = vars(args)
    diff_dict = dict()
    for key, val in args_dict.items():
        setattr(cfg, key, val)
        if key in default_cfg and val != default_cfg[key]:
            diff_dict[key] = val
        elif key not in default_cfg:
            diff_dict[key] = val

    if len(diff_dict) > 0:
        setattr(cfg, 'diff_cfg', diff_dict)
    return args
