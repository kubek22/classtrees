def test():
    """
    My documentation
    """
    print(":)")


def myfun(*args, **kwargs):
    """
    My documentation
    """
    print(args)
    print(kwargs)


# myfun(1, 2, 3, a=4, b=5)
# // Output:
# (1, 2, 3)
# {'a': 4, 'b': 5}