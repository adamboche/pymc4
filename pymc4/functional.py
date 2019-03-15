import inspect
import itertools

import attr
import tensorflow as tf
from tensorflow_probability import edward2
import sympy


_names = itertools.count()


def next_name():
    value = next(_names)
    return f"_SymbolCounter({value})"


ScalarInteger = lambda: sympy.Symbol(next_name())
ScalarFloat = lambda: sympy.Symbol(next_name())


class ExplicitReprSymbol(sympy.Symbol):
    def __repr__(self):
        return "<Symbol {}>".format(super().__repr__())


@attr.s(repr=False)
class Expr:
    formula = attr.ib()

    def __repr__(self):
        return f"<{type(self).__name__} {self.formula}>"


def find_symbol_name(model_class, symbol):
    for k, v in model_class.__dict__.items():
        if symbol == v:
            return k
    raise ValueError(f"Symbol {symbol} not found")


def make_random_variable(model_class, name, random_variable_template):
    symbolic_expressions = {
        k: ExplicitReprSymbol(find_symbol_name(model_class, v))
        for k, v in attr.asdict(random_variable_template).items()
        if isinstance(v, sympy.Expr)
    }
    return lambda: attr.evolve(random_variable_template, **symbolic_expressions)


def make_symbol(name, value):
    return attr.ib()


def make_expr(model_class, name, value):
    symbol_to_name = {
        symbol: name
        for name, symbol in model_class.__dict__.items()
        if symbol in value.free_symbols
    }
    return attr.ib(factory=lambda: Expr(value.subs(symbol_to_name)))


@attr.s
class Observed:
    model = attr.ib()
    data = attr.ib()


def symbolic(cls):
    these = {}
    for k, v in cls.__dict__.items():
        if not isinstance(v, (sympy.Expr, sympy.Symbol)):
            continue
        if isinstance(v, RandomVariableTemplate):
            these[k] = attr.ib(factory=make_random_variable(cls, k, v))
        elif isinstance(v, sympy.Symbol):
            these[k] = make_symbol(k, v)
        elif isinstance(v, sympy.Expr):
            these[k] = make_expr(cls, k, v)

    return attr.make_class(cls.__name__, these)


def observe(model, **data):
    Dataset = attr.make_class("Dataset", list(data.keys()))
    return Observed(model, Dataset(**data))


class RandomVariableTemplate:
    pass


@attr.s(hash=True)
class NormalRV(RandomVariableTemplate, sympy.Symbol):
    loc = attr.ib(default=0.0)
    scale = attr.ib(default=1.0)
    distribution_name = "Normal"

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, next_name())

        return instance


@symbolic
class SchoolsModel:
    num_schools = ScalarInteger()
    sigma = ScalarFloat()
    avg_effect = NormalRV(loc=0.0, scale=10.0)
    avg_stddev = NormalRV(loc=5.0, scale=1.0)
    school_effects_standard = NormalRV()
    school_effects = avg_effect + sympy.exp(avg_stddev) * sigma
    treatment_effects = NormalRV(loc=school_effects, scale=sigma)


inst = SchoolsModel(num_schools=8, sigma=1.0)

print(inst)
# SchoolsModel(
#     num_schools=8,
#     sigma=1.0,
#     avg_effect=NormalRV(loc=0.0, scale=10.0),
#     avg_stddev=NormalRV(loc=5.0, scale=1.0),
#     school_effects_standard=NormalRV(loc=0.0, scale=1.0),
#     school_effects=<Expr avg_effect + sigma*exp(avg_stddev)>,
#     treatment_effects=NormalRV(loc=<Symbol school_effects>, scale=<Symbol sigma>)
# )

observed = observe(inst, treatment_effects=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

print(observed)
# Observed(
#     model=SchoolsModel(
#         num_schools=8,
#         sigma=1.0,
#         avg_effect=NormalRV(loc=0.0, scale=10.0),
#         avg_stddev=NormalRV(loc=5.0, scale=1.0),
#         school_effects_standard=NormalRV(loc=0.0, scale=1.0),
#         school_effects=<Expr avg_effect + sigma*exp(avg_stddev)>,
#         treatment_effects=NormalRV(loc=<Symbol school_effects>, scale=<Symbol sigma>),
#     ),
#     data=Dataset(treatment_effects=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# )
