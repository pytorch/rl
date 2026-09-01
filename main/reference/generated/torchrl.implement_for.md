# implement_for

*class*torchrl.implement_for(*module_name: str | Callable[[], Any]*, *from_version: str | None = None*, *to_version: str | None = None*, ***, *class_method: bool = False*, *compilable: bool = False*)[[source]](../../_modules/pyvers/implement_for.html#implement_for)

A version decorator that checks version compatibility and implements functions.

If specified module is missing or there is no fitting implementation, call of
the decorated function will lead to the explicit error.
In case of intersected ranges, last fitting implementation is used.

This wrapper also works to implement different backends for a same function
(eg. gym vs gymnasium, numpy vs jax-numpy etc).

Parameters:

- **module_name** (*str**or**callable*) - version is checked for the module with this
name (e.g. "gym"). If a callable is provided, it should return the
module.
- **from_version** - version from which implementation is compatible.
Can be open (None).
- **to_version** - version from which implementation is no longer compatible.
Can be open (None).

Keyword Arguments:

- **class_method** (*bool**,**optional*) - if `True`, the function will be written
as a class method. Defaults to `False`.
- **compilable** (*bool**,**optional*) - If `False`, the module import happens
only on the first call to the wrapped function. If `True`, the
module import happens when the wrapped function is initialized.
Defaults to `False`.

Examples

Traditional API (requires `# noqa: F811` on redefinitions):

```
>>> @implement_for("gym", "0.13", "0.14")
... def fun(self, x):
... # Older gym versions will return x + 1
... return x + 1
...
>>> @implement_for("gym", "0.14", "0.23")
... def fun(self, x): # noqa: F811
... # More recent gym versions will return x + 2
... return x + 2
```

This indicates that the function is compatible with gym 0.13+,
but doesn't with gym 0.14+.

Register API (recommended, no `# noqa` needed):

The decorated function has a `.register()` method similar to
`functools.singledispatch`. Use `_` as the function name for
registered implementations to avoid linter warnings:

```
>>> @implement_for("numpy")
... def process_array(arr):
... '''Process array with version-specific implementation.'''
... raise NotImplementedError("No matching implementation")
...
>>> @process_array.register(from_version=None, to_version="2.0.0")
... def _(arr):
... # numpy < 2.0 implementation
... return arr * 2
...
>>> @process_array.register(from_version="2.0.0")
... def _(arr):
... # numpy >= 2.0 implementation
... return arr * 3
```

*static*get_class_that_defined_method(*f: Callable*) → Any | None[[source]](../../_modules/pyvers/implement_for.html#implement_for.get_class_that_defined_method)

Returns the class of a method, if it is defined, and None otherwise.

*classmethod*import_module(*module_name: str | Callable[[], Any]*) → str[[source]](../../_modules/pyvers/implement_for.html#implement_for.import_module)

Imports module and returns its version.

module_set() → None[[source]](../../_modules/pyvers/implement_for.html#implement_for.module_set)

Sets the function in its module, if it exists already.

*classmethod*reset(*setters_dict: dict[str, implement_for] | None = None*) → None[[source]](../../_modules/pyvers/implement_for.html#implement_for.reset)

Resets the setters in setter_dict.

Parameters:

**setters_dict** - A copy of implementations. We iterate through its values
and call `module_set()` for each.