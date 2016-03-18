

def change_doc(other_func, add_warning = True, keep_new_doc = True):
    '''This decorator changes the docstring to that of another function
    Arguments
    ---------
    add_warning: bool
        if a warning is printed
    keep_new_doc: bool
        If the newly defined docstring is added to the changed one
    '''

    doc_other = other_func.__doc__
    def wrapper(func):
        def inner(*args, **kwargs):
            wrapped_func = func(*args, **kwargs)
#             func.__doc__ = other_func.__doc__
            return wrapped_func
        if keep_new_doc and func.__doc__:
            note = """Note from wrapper:
"""
            wrapper_doc = func.__doc__
            note_II = 'Original doc string:\n'
            doc = note + wrapper_doc + '\n\n' + note_II
        else:
            doc = ''
        if add_warning:
            txt = 'This is the docstring of a function that has been decorated/composed. Function might behave wired.\n\n'
            doc = txt + doc
        inner.__doc__ = doc + doc_other
        return inner
    return wrapper