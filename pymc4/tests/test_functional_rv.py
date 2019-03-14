import attr
import tensorflow as tf
from tensorflow_probability import edward2 as ed
import numpy as np


@attr.s
class Bernoulli:
    mu = attr.ib()
    shape = attr.ib(factory=tuple)


@attr.s
class Normal:
    loc = attr.ib(default=0.0)
    scale = attr.ib(1.0)


num_schools = 8


def test_bernoulli_rv():
    pass


if False:
    model = pm.Model(num_schools=J, y=y, sigma=sigma)

    @model.define
    def process(cfg):
        mu = ed.Normal(loc=0.0, scale=5.0, name="mu")  # `mu` above
        # Due to the lack of HalfCauchy distribution.
        log_tau = ed.Normal(loc=5.0, scale=1.0, name="log_tau")  # `log(tau)` above
        theta_prime = ed.Normal(
            loc=tf.zeros(cfg.num_schools), scale=tf.ones(cfg.num_schools), name="theta_prime"
        )  # `theta_prime` above
        theta = mu + tf.exp(log_tau) * theta_prime  # `theta` above
        y = ed.Normal(loc=theta, scale=np.float32(cfg.sigma), name="y")  # `y` above

        return y


# class EightSchoolsModel:
#     avg_effect = Normal(loc=0.0, scale=10.0)  # `mu` above
#     avg_stddev = Normal(loc=5.0, scale=1.0)  # `log(tau)` above
#     school_effects_standard = Normal(
#         tf.zeros(num_schools), scale=tf.ones(num_schools)
#     )  # `theta_prime` above
#     # This needs + and * to be implemented, which may rely on tensorflow.
#     # But possibly edward2 isn't using a context at this stage.
#     school_effects = avg_effect + tf.exp(avg_stddev) * school_effects_standard  # `theta` above
#     treatment_effects = Normal(loc=school_effects)  # `y` above


class EightSchoolsModel:
    def __init__(self, num_schools, avg_effect, avg_stddev, sigma):
        self.avg_effect = Normal(loc=0.0, scale=10.0)  # `mu` above
        self.avg_stddev = Normal(loc=5.0, scale=1.0)  # `log(tau)` above
        self.school_effects_standard = Normal(
            tf.zeros(num_schools), scale=tf.ones(num_schools)
        )  # `theta_prime` above
        self.school_effects = avg_effect + tf.exp(avg_stddev) * sigma  # `theta` above
        self.treatment_effects = Normal(loc=self.school_effects, scale=sigma)  # `y` above


ScalarFloat = attr.ib
ScalarInteger = attr.ib

# We build a model declaratively as a class with a decorator.


@model
class SchoolsModel:
    sigma = ScalarFloat()
    num_schools = ScalarInteger()
    avg_effect = Normal(loc=0.0, scale=10.0)  # `mu` above
    avg_stddev = Normal(loc=5.0, scale=1.0)  # `log(tau)` above
    school_effects_standard = Normal(tf.zeros(num_schools), scale=tf.ones(num_schools))
    school_effects = avg_effect + tf.exp(avg_stddev) * sigma  # `theta` above
    treatment_effects = Normal(loc=school_effects, scale=sigma)  # `y` above


# The distributions could optionally take a shape parameter instead of needing scale=tf.ones(num_schools).

# Each distribution instantiation in the class definition produces a symbolic expression. The `school_effects` formula produces a symbolic expression. @model replaces the CountingAttr equivalent with a new attribute having the name of the attribute inside it, so that the formula can know all its terms' names.

# Non-distribution quantities must be assigned explicitly at __init__. The other attributes may be left unfixed. So we instantiate it as `SchoolsModel(sigma=1.0, num_schools=8)` and get an object with repr like this, though perhaps assigning the numbers instead of the variable names (num_schools) in the repr:

# SchoolsModel(
#     sigma=1.0,
#     num_schools=8,
#     avg_effect=Normal(loc=0.0, scale=10.0),
#     avg_stddev=Normal(loc=5.0, scale=1.0),
#     school_effects_standard=Normal(tf.zeros(num_schools), scale=tf.ones(num_schools)),
#     school_effects=avg_effect + tf.exp(avg_stddev) * sigma,
#     treatment_effects=Normal(loc=school_effects, scale=sigma),
# )

# observe(**data) or optionally __init__(**data) assigns observed data to the variables. observe() is like attr.evolve() but it combines the value of each observed variable with its prior into a new instance attribute.

# observed_model = observe(treatment_effects=[1., 2., 3., 4., 5., 6., 7., 8.])

# Now the repr is

# SchoolsModel(
#     sigma=1.0,
#     num_schools=8,
#     avg_effect=Normal(loc=0.0, scale=10.0),
#     avg_stddev=Normal(loc=5.0, scale=1.0),
#     school_effects_standard=Normal(tf.zeros(num_schools), scale=tf.ones(num_schools)),
#     school_effects=avg_effect + tf.exp(avg_stddev) * sigma,
#     treatment_effects=Observed(
#         prior=Normal(loc=school_effects, scale=sigma),
#         data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
#     ),
# )

# Sample from the observed instance:

# trace = sample(observed_model)

# Summarize:

# summary_table(trace)
# summary_plot(trace)

print("defined")
esm = EightSchoolsModel(num_schools=8, avg_effect=1.0, avg_stddev=1.0, sigma=1.0)
print("done")
print(esm)
