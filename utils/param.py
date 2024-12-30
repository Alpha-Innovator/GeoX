import logging
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

def detach_and_clone_param(param, ignore_status=False, name=None):
    """
    Detach and clone a parameter, optionally handling DeepSpeed zero optimization status.
    
    Args:
        param: The parameter to detach and clone.
        ignore_status (bool): Whether to ignore the DeepSpeed zero optimization status.
        name (str): The name of the parameter (for logging purposes).
    
    Returns:
        A detached and cloned copy of the parameter.
    """
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_parameters_by_keys(named_params, keys_to_match):
    """
    Get parameters from named_params that match any key in keys_to_match.
    
    Args:
        named_params (dict): A dictionary of named parameters.
        keys_to_match (list): A list of keys to match against the parameter names.
    
    Returns:
        A dictionary of matching parameters.
    """
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: detach_and_clone_param(v, ignore_status=True) for k, v in to_return.items()}
    return to_return