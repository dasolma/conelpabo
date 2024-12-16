import phm_framework


def flat_dict(_dict):
    keys = list(_dict.keys())

    if len(keys) > 0:
        k = keys[0]
        v = _dict[k]
        del _dict[k]

        if isinstance(v, dict):
            r = {f"{k}__{sk}": sv for sk, sv in v.items()}

        else:
            r = {k: v}

        r.update(flat_dict(_dict))
        return r
    else:
        return {}


def get_model_creator(net_name):

    net_creator_func = f"create_model"
    if isinstance(net_name, list):
        net_name = net_name[0]
    net_module = getattr(getattr(phm_framework, 'nets'), net_name)
    creator = getattr(net_module, net_creator_func)

    return creator