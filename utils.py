def difference(original, other):
    try:
        return [element for element in original if element not in other]
    except TypeError:
        return original.remove(other)