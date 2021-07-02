import os
import traceback
from typing import Union

from tqdm import tqdm

from src.utils.config import Config

bcolors = {'PINK': '\033[95m',
           'BLUE': '\033[94m',
           'CYAN': '\033[96m',
           'GREEN': '\033[92m',
           'YELLOW': '\033[93m',
           'RED': '\033[91m',}


def pretty_string(message: str, color=None, bold=False, underline=False):
    """
    add color and effects to string
    :param message:
    :param color:
    :param bold:
    :param underline:
    :return:
    """
    ou = message
    if color:
        ou = bcolors[color] + message + '\033[0m'
    if bold:
        ou = '\033[1m' + ou + '\033[0m'
    if underline:
        ou = '\033[4m' + ou + '\033[0m'
    return ou


def pyout(message: Union[None,str] = None):
    """
    Print message preceded by traceback. I use this method to prevent rogue "print" statements
    during debugging
    :param message:
    :return:
    """
    if Config.verbose:
        trace = traceback.extract_stack()[-2]

        fname = trace.filename.replace(os.path.abspath(os.curdir), "...")

        trace = f"{fname}: {trace.name}(...) - ln{trace.lineno}"

        tqdm.write(pretty_string(trace, 'PINK', bold=True))
        if message is not None:
            tqdm.write(message)
