import functools
import inspect
import warnings
from typing import Any, Callable, Tuple, Union, Optional
import numpy as np
from utils import __float_dtype__
from _thread import RLock

warnings.filterwarnings("default")


class CheckError(Exception):
    pass


class Checker:
    """
    Base class for parameter and return value checkers
    
    Attributes:
        checked_objects: Set of object IDs that have already been checked
        arg_checker: Function to validate and modify parameters
        return_checker: Function to validate and modify return values
    """
    checked_objects = dict()
    _checked_lock = RLock()
    def __init__(self):
        if not (hasattr(self, 'arg_checker') and callable(self.arg_checker)):
            self.arg_checker = None
        if not (hasattr(self, 'return_checker') and callable(self.return_checker)):
            self.return_checker = None

    def clear_checked(self, val):
        with self._checked_lock:
            idval = id(val)
            stack = self.checked_objects.get(idval)
            if not stack:
                return
            stack.pop()
            if len(stack) == 0:
                self.checked_objects.pop(idval, None)

    def add_checked(self, val):
        with self._checked_lock:
            if val is None:
                return
            idval = id(val)
            if idval not in self.checked_objects:
                self.checked_objects[idval] = [type(self)]
            else:
                self.checked_objects[idval].append(type(self))

    def is_checked(self, val):
        with self._checked_lock:
            if val is None:
                return True
            idval = id(val)
            stack = self.checked_objects.get(idval)
            if not stack:
                return False
            return issubclass(stack[-1], type(self))

    def check_warning(self, text, funcinfo=None, warn_type=RuntimeWarning):
        if funcinfo is not None:
            text = f"{text} (Function `{funcinfo.get('funcname', '')}` in `{funcinfo.get('funcfile', '')}:{funcinfo.get('funcline', '')}`)"
        warnings.warn(text, warn_type)

    def _check_parameters(self, bound_args, params_to_check, info=None, param_positions=None) -> tuple:
        """
        Check function parameters using the provided checker
        
        Args:
            bound_args: Bound arguments from function signature
            params_to_check: Parameter inspection configuration
            checker: Checker instance with check_argument method
        
        Returns:
            Tuple of (modified_args, modified_kwargs)
        """
        added_list = []
        if not params_to_check or not self.arg_checker:
            return bound_args.args, bound_args.kwargs, added_list

        param_names = list(bound_args.arguments.keys()) if params_to_check is True else params_to_check
        modified_args = list(bound_args.args)
        modified_kwargs = dict(bound_args.kwargs)
        if param_positions is None:
            sig_params = list(bound_args.signature.parameters.keys())
            param_positions = {name: idx for idx, name in enumerate(sig_params)}

        for param_name in param_names:
            if param_name in bound_args.arguments:
                arg_value = bound_args.arguments[param_name]
                if not self.is_checked(arg_value):
                    new_value = self.arg_checker(arg_value, param_name=param_name, info=info)
                    if new_value is not arg_value:
                        if param_name in bound_args.kwargs:
                            modified_kwargs[param_name] = new_value
                        else:
                            param_index = param_positions.get(param_name)
                            if param_index is not None and param_index < len(modified_args):
                                modified_args[param_index] = new_value
                    self.add_checked(new_value)
                    added_list.append(new_value)
            else:
                self.check_warning(f"Parameter '{param_name}' not found in function arguments.", info)

        return tuple(modified_args), modified_kwargs, added_list


    def _check_return_value(self, result, returns, info=None) -> Any:
        """
        Check and possibly modify function return value using the provided checker
        
        Args:
            result: The original return value
            returns: Return value inspection configuration
            checker: Checker instance with check_return_value method
        
        Returns:
            The checked (and possibly modified) return value
        """
        added_list = []
        if not returns or result is None or not self.return_checker:
            return result, added_list

        if isinstance(result, tuple):
            indices = returns if isinstance(returns, tuple) else range(len(result))
            results = list(result)
            modified = False
            
            for idx in indices:
                if idx < len(results):
                    return_item = results[idx]
                    if not self.is_checked(return_item):
                        new_value = self.return_checker(return_item, idx=idx, info=info)
                        
                        if new_value is not return_item:
                            results[idx] = new_value
                            modified = True

                        self.add_checked(new_value)
                        added_list.append(new_value)
            return (tuple(results) if modified else result), added_list
        else:
            if not self.is_checked(result):
                result = self.return_checker(result, info=info)
                self.add_checked(result)
                added_list.append(result)
            return result, added_list


class AsarrayChecker(Checker):
    def __init__(self):
        super().__init__()

    def arg_checker(self, value, param_name=None, info=None, **kwargs):
        """Convert input to numpy array"""
        try:
            arr = np.asarray(value, dtype=__float_dtype__)
        except ValueError:
            raise CheckError(f"Input Param({param_name}) must be a number or an array-like of numbers.")
        return arr

    def return_checker(self, value, idx=0, info=None, **kwargs):
        """Convert output to numpy array"""
        try:
            arr = np.asarray(value, dtype=__float_dtype__)
        except ValueError:
            raise CheckError(f"Out Param({idx}) must be a number or an array-like of numbers.")
        return arr

class ScaleoneChecker(AsarrayChecker):
    def __init__(self):
        super().__init__()
    
    def _scaleone(self, value):
        return bool(np.all(np.logical_and(0. <= value, value <= 1.)))

    def arg_checker(self, value, param_name=None, info=None, **kwargs):
        value = AsarrayChecker.arg_checker(self, value, param_name)
        if self._scaleone(value):
            return value
        raise CheckError(f"Input Param({param_name}) must be in the range [0, 1].")

    def return_checker(self, value, idx=0, info=None, **kwargs):
        value = AsarrayChecker.return_checker(self, value, idx)
        if not self._scaleone(value):
            self.check_warning(f"Out Param({idx}) is not in the range [0, 1], it will be scaled to [0, 1].", info)
            value = np.clip(value, 0., 1.)
        return value

class ProbabilityChecker(ScaleoneChecker):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def _sumone(self, value):
        close = np.isclose(
            np.sum(value, axis=self.axis),
            1.0,
            atol=np.finfo(value.dtype).eps if isinstance(value, np.ndarray) else 1e-6,
        )
        return bool(np.all(close))

    def arg_checker(self, value, param_name=None, info=None, **kwargs):
        value = ScaleoneChecker.arg_checker(self, value, param_name)
        if self._sumone(value):
            return value
        axis_text = f" in axis({self.axis})" if self.axis is not None else ""
        raise CheckError(f"Input Param({param_name}) must be a valid probability distribution{axis_text}.")

    def return_checker(self, value, idx=0, info=None, **kwargs):
        """Ensure the output is a valid probability distribution"""        
        value = AsarrayChecker.return_checker(self, value, idx)
        if not (self._sumone(value) and self._scaleone(value)):
            value = value / np.sum(value, axis=self.axis, keepdims=True)
            value = np.clip(value, 0., 1.)
            axis_text = f" in axis({self.axis})" if self.axis is not None else ""
            self.check_warning(f"Out Param({idx}) is not a valid probability distribution{axis_text}, it will be normalized.", info)
        return value


def inspector(
    args_to_check: Union[bool, Tuple[str]] = True, 
    returns_to_check: Union[bool, Tuple[int, ...]] = False,
    checker: Optional[Checker] = None
):
    """
    Decorator factory for parameter and return value inspection
    
    Args:
        args_to_check: Parameter inspection option
            - True: Inspect all parameters
            - False: Skip parameter inspection
            - ('x', 'y'): Inspect specific parameters
        returns_to_check: Return value inspection option
            - True: Inspect all return values
            - False: Skip return inspection
            - (0, 2): Inspect specific indices in return tuple
        checker: Checker instance with check_argument and check_return_value methods
    """
    if isinstance(args_to_check, str) and not isinstance(args_to_check, bool):
        args_to_check = (args_to_check,)
    if isinstance(returns_to_check, int) and not isinstance(returns_to_check, bool):
        returns_to_check = (returns_to_check,)
    def decorator(func: Callable) -> Callable:
        if checker is None:
            return func

        sig = inspect.signature(func)
        param_positions = {name: idx for idx, name in enumerate(sig.parameters.keys())}

        info = {
            'funcfile': None,
            'funcline': None,
            'funcname': func.__name__,
        }
        try:
            info['funcfile'] = inspect.getsourcefile(func)
        except (OSError, TypeError):
            pass
        try:
            info['funcline'] = inspect.getsourcelines(func)[1]
        except (OSError, TypeError):
            pass

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            added_args, added_returns = [], []
            try:
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                modified_args, modified_kwargs, added_args = checker._check_parameters(
                    bound_args,
                    args_to_check,
                    info=info,
                    param_positions=param_positions,
                )
                result = func(*modified_args, **modified_kwargs)
                result, added_returns = checker._check_return_value(result, returns_to_check, info=info)
                
                return result
                
            finally:
                for v in added_args:
                    checker.clear_checked(v)
                for v in added_returns:
                    checker.clear_checked(v)

        return wrapper
    return decorator
