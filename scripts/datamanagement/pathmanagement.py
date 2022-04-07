from scripts.constants import *
import datetime
import glob


def create_directories(path: str):
    """create all directories in the path
    that do not exist and return path"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def replace_symbols(var: str, to_replace: str, replacement: str):
    """replace all required symbols in the string with desired"""
    for symbol in to_replace:
        var = var.replace(symbol, replacement)
    return var


def gettimestamp():
    """date and time to a string"""
    s = str(datetime.datetime.today())
    return replace_symbols(s, to_replace=' -:.', replacement='_')


def find_all(path, pattern):
    """return all paths with pattern"""
    return glob.glob(f"{path}/**/*{pattern}*", recursive=True)
