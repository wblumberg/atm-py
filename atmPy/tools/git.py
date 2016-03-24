import subprocess as _sp

def current_commit():
    return _sp.check_output(["git", "describe", "--always"]).strip()