"""!
@brief Pytorch abstract dataset class for inheritance.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

from abc import abstractmethod
import inspect


class Dataset:
    @abstractmethod
    def get_arg_and_check_validness(self,
                                    key,
                                    choices=None,
                                    known_type=None,
                                    extra_lambda_checks=None):
        try:
            value = self.kwargs[key]
        except Exception as e:
            print(e)
            raise KeyError("Argument: <{}> does not exist in pytorch "
                           "dataloader keyword arguments".format(key))

        if known_type is not None:
            if not isinstance(value, known_type):
                raise TypeError("Value: <{}> for key: <{}> is not an "
                                "instance of "
                                "the known selected type: <{}>"
                                "".format(value, key, known_type))

        if choices is not None:
            if isinstance(value, list):
                if not all([v in choices for v in value]):
                    raise ValueError("Values: <{}> for key: <{}>  "
                                     "contain elements in a"
                                     "regime of non appropriate "
                                     "choices instead of: <{}>"
                                     "".format(value, key, choices))
            else:
                if value not in choices:
                    raise ValueError("Value: <{}> for key: <{}> is "
                                     "not in the "
                                     "regime of the appropriate "
                                     "choices: <{}>"
                                     "".format(value, key, choices))

        if extra_lambda_checks is not None:
            all_checks_passed = all([f(value)
                                     for f in extra_lambda_checks])
            if not all_checks_passed:
                raise ValueError(
                    "Value(s): <{}> for key: <{}>  "
                    "does/do not fulfill the predefined checks: "
                    "<{}>".format(value, key,
                    [inspect.getsourcelines(c)[0][0].strip()
                     for c in extra_lambda_checks
                     if not c(value)]))
        return value
