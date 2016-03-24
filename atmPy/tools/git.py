import subprocess as _sp
import os as _os


def current_commit():
    root = _os.path.split(__file__)[0].split('/')[:-2]
    root = '/'.join(root)
    # print(root)
    out = _sp.check_output(["git", "--git-dir=%s/.git" % root, "--work-tree=%s/" % root, "describe","--always"]).strip()
    return out
    # return _sp.check_output(["git", "--git-dir=/Users/htelg/prog/atm-py/.git", "--work-tree=/Users/htelg/prog/atm-py/", "describe", "--always"]).strip()
    #
    # this_dir = _os.path.split(__file__)[0]
    # print('this_dir', this_dir)
    # main_dir = _os.path.split(this_dir)[0]
    # print('main_dir', main_dir)
    # import pdb as _pdb
    # _pdb.set_trace()