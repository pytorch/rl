# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functorch


def get_params_of_module(module, cf, p, b):
    split_name_dict = make_split_names_dict(cf.split_names, p, b)
    names, values = _get_params_of_module(module, cf.stateless_model, split_name_dict)
    S = set(values)
    S_param = S.intersection(set(p))
    S_buffer = S.intersection(set(b))

    name_p_dict = {_p: _name for _p, _name in zip(values, names)}
    param_names, params = zip(*[(name_p_dict[_p], _p) for _p in p if _p in S_param])
    buffer_names, buffers = zip(*[(name_p_dict[_b], _b) for _b in b if _b in S_buffer])

    fmodule = functorch.FunctionalModuleWithBuffers(module, param_names, buffer_names)
    return fmodule, params, buffers


def _get_params_of_module(module, target, split_name_dict):
    if target is module:
        return _get_params(split_name_dict)
    else:
        found = False
        for name in split_name_dict:
            sub_target = getattr(target, name)
            sub_split_name_dict = split_name_dict[name]
            if isinstance(sub_split_name_dict, dict):
                out = _get_params_of_module(module, sub_target, sub_split_name_dict)
                if out:
                    return out
        return found


def _get_params(dictionary):
    out = []
    for key, value in dictionary.items():
        if not isinstance(value, dict):
            out.append((key, value))
        else:
            _out = [
                (".".join([key, _key]), _val)
                for (_key, _val) in zip(*_get_params(value))
            ]
            out += _out
    return tuple(zip(*out))


def get_item(d, name, p):
    _d = d[name[0]]
    if isinstance(_d, dict):
        get_item(_d, name[1:], p)
    else:
        p.append(_d)


def populate_params(split_names, d, p=None):
    if p is None:
        p = []
    for name in split_names:
        get_item(d, name, p)
    return p


class apply_to_class:
    def __init__(self, layer_type):
        self.layer_type = layer_type

    def __call__(self, func):
        def new_func(cf, p, b, **kwargs):
            split_name_dict = make_split_names_dict(cf.split_names, p, b)
            d = self.dispatch_to_layers(
                func, self.layer_type, split_name_dict, cf.stateless_model
            )
            new_p = populate_params(cf.split_names, d)
            new_p, new_b = new_p[: len(p)], new_p[len(p) :]
            return new_p, new_b

        return new_func

    @staticmethod
    def dispatch_to_layers(func, layer_type, split_name_dict, cf):
        if isinstance(cf, layer_type):
            split_name_dict.update(func(cf, split_name_dict))

        for layer_name in split_name_dict:
            layer_or_param = getattr(cf, layer_name)
            if isinstance(layer_or_param, nn.Module):
                split_name_dict[layer_name] = apply_to_class.dispatch_to_layers(
                    func,
                    layer_type,
                    split_name_dict[layer_name],
                    layer_or_param,
                )
        return split_name_dict


def make_split_names_dict(split_names, p, b=[], split_name_dict=None):
    if split_name_dict is None:
        split_name_dict = dict()

    _firsts = dict()
    for name, param in zip(split_names, list(p) + list(b)):
        if len(name) > 1:
            layer_list = _firsts.get(name[0], [])
            layer_list.append((name[1:], param))
            _firsts[name[0]] = layer_list
        else:
            split_name_dict[name[0]] = param
    for key in _firsts:
        _names, _params = zip(*_firsts[key])
        _dict = make_split_names_dict(_names, _params)
        split_name_dict[key] = _dict
    return split_name_dict


def get_submodule_functional(module, cf):
    p = [i for i, _ in enumerate(cf.param_names)]
    b = [len(p) + i for i, _ in enumerate(cf.buffer_names)]
    split_name_dict = make_split_names_dict(cf.split_names, p, b)
    names, values = _get_params_of_module(module, cf.stateless_model, split_name_dict)
    S = set(values)
    S_param = S.intersection(set(p))
    S_buffer = S.intersection(set(b))

    name_p_dict = {_p: _name for _p, _name in zip(values, names)}
    param_names, params = zip(*[(name_p_dict[_p], _p) for _p in p if _p in S_param])
    if len(S_buffer):
        buffer_names, _ = zip(*[(name_p_dict[_b], _b) for _b in b if _b in S_buffer])
    else:
        buffer_names, _ = tuple(), tuple()

    fmodule = functorch.FunctionalModuleWithBuffers(module, param_names, buffer_names)
    return fmodule
