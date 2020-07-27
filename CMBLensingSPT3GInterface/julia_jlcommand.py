import platform
import sys

from julia import Main

def jl(src, locals=None, globals=None):
    """ 
    Execute some Julia code and return the answer. The Julia code is executed in
    the top-level of Julia's Main module.
    
    You can interpolate Python values into the Julia expression in the same way
    as with `%julia` magic for IPython, and scoping behaves the same way as well.
    
    See: https://pyjulia.readthedocs.io/en/latest/usage.html#ipython-magic
    """
    
    if locals  is None: locals  = get_caller_locals()
    if globals is None: globals = get_caller_globals()

    src = unicode(src)
    return_value = "nothing" if src.strip().endswith(";") else ""

    return Main.eval(
        """
        _PyJuliaHelper.@prepare_for_pyjulia_call begin
            begin %s end
            %s
        end
        """
        % (src, return_value)
    )(globals, locals)


def safe_getframe(i=0):
    msg = 'provide locals/globals explicitly: `jl("...", locals(), globals())`'
    if platform.python_implementation()!='CPython':
        raise Exception("On non-CPython implementations, you must "+msg)
    else:
        try:
            return sys._getframe(i)
        except:
            raise Exception("Unable to get caller frame. Please "+msg)
        

def get_caller_globals():
    return safe_getframe(3).f_globals


def get_caller_locals():
    
    # if the caller frame is a comprehension, we keep recursing up accumulating
    # locals until we hit a non-comprehension frame, since these variables may not be in
    # scope unless they were explicitly used in Python code
    # 
    # on CPython, comprehension frames are designated by containing a ".0" variable
    # (see https://docs.python.org/3/library/inspect.html#inspect.Parameter.name)
    # 
    # https://stackoverflow.com/questions/54761252/python3-list-comprehensions-and-stack-frames
    # is a nice discussion of the difference between frame and scope which is
    # basically the issue at hand here.

    frame = safe_getframe(3)
    all_f_locals = [frame.f_locals.copy()]
    while ".0" in frame.f_locals:
        all_f_locals[-1].pop(".0")
        frame = frame.f_back
        all_f_locals.append(frame.f_locals)
    caller_locals = {}
    all_f_locals.reverse()
    for f_locals in all_f_locals:
        caller_locals.update(f_locals)
    return caller_locals


# don't know what this does but I'm just gonna follow pyjulia's example
# https://github.com/JuliaPy/pyjulia/blob/0c0d74f72c473c60ea530b8f3c105aba6c903f27/src/julia/magic.py#L31-L34
try:
    unicode
except NameError:
    unicode = str
