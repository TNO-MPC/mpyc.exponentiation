"""
Unit tests for the exponent.py module.
"""

import math
from functools import partial
from typing import List, Type, TypeVar

import pytest
from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint, SecureInteger

from tno.mpc.mpyc.exponentiation import secure_pow
from tno.mpc.mpyc.exponentiation.exponent import (
    _secure_pow_fractional,
    maximal_exponent,
    minimal_exponent,
)
from tno.mpc.mpyc.exponentiation.utils import convert_to_secfxp, convert_to_secint

pytestmark = [
    pytest.mark.asyncio,
]

SecureTypes = TypeVar("SecureTypes", Type[SecureFixedPoint], Type[SecureInteger])

# sectypes for general use
secint: Type[SecureInteger] = mpc.SecInt()
secint16: Type[SecureInteger] = mpc.SecInt(16)
secfxp: Type[SecureFixedPoint] = mpc.SecFxp()
secfxp16: Type[SecureFixedPoint] = mpc.SecFxp(16)

secnums = [secint, secfxp]
secnums_ext = [secint, secint16, secfxp, secfxp16]

DEFAULT_BASE = 2
DEFAULT_BIT_PRECISION = secfxp.frac_length // 2
pytest_approx_default = partial(
    pytest.approx,
    abs=2 ** -(DEFAULT_BIT_PRECISION // 2),
    rel=2 ** -(DEFAULT_BIT_PRECISION // 2 - 1),
)
maximal_exponent_secint = maximal_exponent(base=DEFAULT_BASE, stype=secint)
minimal_exponent_secint = max(
    minimal_exponent(base=DEFAULT_BASE, stype=secint), -maximal_exponent_secint
)
maximal_exponent_secfxp = maximal_exponent(base=DEFAULT_BASE, stype=secfxp)
minimal_exponent_secfxp = max(
    minimal_exponent(base=DEFAULT_BASE, stype=secfxp), -maximal_exponent_secfxp
)


def floor_to_odd_integer(x: float) -> int:
    """
    Round down to the nearest odd number.

    >>> floor_to_odd_integer(3)
    3
    >>> floor_to_odd_integer(4)
    3

    :param x: Value to be rounded.
    :return: Nearest lower odd number no larger than the provided value.
    """
    x_floor = math.floor(x)
    return x_floor if x_floor % 2 == 1 else x_floor - 1


def truncate_exponented_value(
    value: float,
    lower_bound: float,
    lower_fill: float,
    upper_bound: float,
    upper_fill: float,
) -> float:
    r"""
    Perform the plaintext equivalent of the trunc_to_domain operation from the secure_pow function.

    Given an value x, return lower_fill if x < lower_value, upper_fill if x > upper_value and x otherwise.

    :param value: Value to be transformed.
    :param lower_bound: Lower bound of interval where x remains unchanged.
    :param lower_fill: Return value if x is smaller than lower_bound.
    :param upper_bound: Upper bound of interval where x remains unchanged.
    :param upper_fill: Return value if x is larger than upper_bound.
    :return: Truncation of the value in the sense of the description above.
    """
    assert lower_bound <= upper_bound
    if value < lower_bound:
        return lower_fill
    if value > upper_bound:
        return upper_fill
    return value


class TestConvertToSecint:
    """
    Tests for convert_to_secint.
    """

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("secnum", (secfxp,))
    async def test_convert_to_secint_type(secnum: Type[SecureFixedPoint]) -> None:
        """
        Verify type of return value.

        :param secnum: secure number type
        """
        plaintext = 1
        secure_value = mpc.input(secnum(plaintext), 0)
        return_value = convert_to_secint(secure_value)
        assert isinstance(return_value, SecureInteger)

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("secnum", (secfxp, secfxp16))
    async def test_convert_to_secint_list_type(secnum: Type[SecureFixedPoint]) -> None:
        """
        Verify type of return value for lists.

        :param secnum: secure number type
        """
        plaintext_list = [-1, 0, 1]
        secure_list = mpc.input(list(map(secnum, plaintext_list)), 0)
        return_value = convert_to_secint(secure_list)
        assert isinstance(return_value, list)
        assert len(return_value) == len(plaintext_list)
        assert all(isinstance(_, SecureInteger) for _ in return_value)

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("secnum", (secfxp, secfxp16))
    async def test_convert_to_secint_list_outcome(
        secnum: Type[SecureFixedPoint],
    ) -> None:
        """
        Verify output and modulus of conversions from lists of secint and secfxp to lists of secint.

        :param secnum: secure number type
        """
        plaintext_list = [-1, 0, 1]
        secure_list = mpc.input(list(map(secnum, plaintext_list)), 0)
        return_value = convert_to_secint(secure_list)

        for i in range(len(plaintext_list)):
            return_field = return_value[i].field
            x_sec_field = secure_list[i].field
            assert return_field is not None
            assert x_sec_field is not None
            assert return_field.modulus == x_sec_field.modulus
        assert await mpc.output(return_value) == plaintext_list


class TestConvertToSecfxp:
    """
    Tests for convert_to_secfxp.
    """

    @staticmethod
    @pytest.mark.parametrize("secnum", secnums)
    def test_convert_to_secfxp_type(secnum: SecureTypes) -> None:
        """
        Verify type of return value.

        :param secnum: secure number type
        """
        plaintext = 1 if issubclass(secnum, SecureInteger) else 1.5
        secure_value = mpc.input(secnum(plaintext), 0)
        return_value = convert_to_secfxp(secure_value, secfxp)
        assert isinstance(return_value, secfxp)

    @staticmethod
    @pytest.mark.parametrize("secnum", secnums_ext)
    def test_convert_to_secfxp_list_type(secnum: SecureTypes) -> None:
        """
        Verify type of return value for lists.

        :param secnum: secure number type
        """
        plaintext_list: List[float] = (
            [0, 1] if issubclass(secnum, SecureInteger) else [0.0, 1.5]
        )
        secure_list = mpc.input(list(map(secnum, plaintext_list)), 0)
        return_value = convert_to_secfxp(secure_list, secfxp)
        assert isinstance(return_value, list)
        assert len(return_value) == len(plaintext_list)
        assert all(isinstance(_, secfxp) for _ in return_value)

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("secfxp_type", (secfxp, secfxp16))
    @pytest.mark.parametrize("secnum", secnums_ext)
    async def test_convert_to_secfxp_list_outcome(
        secnum: SecureTypes, secfxp_type: Type[SecureFixedPoint]
    ) -> None:
        """
        Verify output and modulus of conversions from lists of secint and secfxp to lists of secfxp.

        :param secnum: secure number type
        :param secfxp_type: secure fixed point type to convert to
        """
        plaintext_list: List[float] = (
            [-1, 0, 1] if issubclass(secnum, SecureInteger) else [-1.5, 0.0, 1.5]
        )
        secure_list = mpc.input(list(map(secnum, plaintext_list)), 0)
        secnum_field = secnum.field
        secfxp_field = secfxp_type.field
        assert secnum_field is not None
        assert secfxp_field is not None
        if (
            secnum.frac_length <= secfxp_type.frac_length
            and secfxp_field.modulus == secnum_field.modulus
        ):
            return_value_fxp = convert_to_secfxp(secure_list, secfxp_type)
            return_value = await mpc.output(return_value_fxp)
            assert return_value == plaintext_list
        else:
            with pytest.raises(AssertionError):
                convert_to_secfxp(secure_list, secfxp_type)

    @staticmethod
    def test_convert_to_secfxp_errors() -> None:
        """
        Verify output of conversions for edge cases.
        """
        with pytest.raises(AssertionError):
            plaintext = 1
            secure_value = mpc.input(secfxp(plaintext), 0)
            convert_to_secfxp(secure_value, secfxp16)


class TestSecurePowFractional:
    """
    Tests for exponentiation b**x where x is non-integral and has modulus of at most 1.
    """

    @staticmethod
    def test_secure_pow_fractional_type() -> None:
        """
        Verify type of return value.
        """
        base = 2
        plaintext_list = [x_i / 10 for x_i in range(-10, 11)]
        x_sec = mpc.input(list(map(secfxp, plaintext_list)), 0)
        return_value = _secure_pow_fractional(base, x_sec)
        assert isinstance(return_value, list)
        assert len(return_value) == len(plaintext_list)
        assert all(isinstance(_, secfxp) for _ in return_value)

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("base", (2, 3))
    async def test_secure_pow_fractional_outcome(base: int) -> None:
        """
        Verify that the secure exponentiation returns the correct result.

        :param base: base of the exponentiation
        """
        plaintext_list = [x_i / 10 for x_i in range(-10, 11)]
        plaintext_list_powed = [base ** xi for xi in plaintext_list]
        secure_list = mpc.input(list(map(secfxp, plaintext_list)), 0)
        bit_precision = secfxp.frac_length // 2
        return_value = _secure_pow_fractional(
            base, secure_list, bit_precision=bit_precision
        )
        return_value = await mpc.output(return_value)
        # TODO: how precise?
        assert return_value == pytest.approx(
            plaintext_list_powed, abs=2 ** -(bit_precision // 2)
        )

    @staticmethod
    @pytest.mark.asyncio
    async def test_secure_pow_fractional_outcome_with_nones() -> None:
        """
        Verify that the secure exponentiation correctly deals with None input.
        These inputs should return a secure 1.
        """
        base = 2
        plaintext_list = [None, -0.5, 0.5, None]
        plaintext_list_powed = [base ** x_i if x_i else 1 for x_i in plaintext_list]
        secure_list = [
            mpc.input(secfxp(x_i), 0) if x_i else None for x_i in plaintext_list
        ]
        return_value = _secure_pow_fractional(base, secure_list)
        return_value = await mpc.output(return_value)
        assert return_value == pytest.approx(plaintext_list_powed, abs=0.5)

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("base", (2, 3, 4))
    @pytest.mark.parametrize("bit_precision", (2, 6, 10))
    async def test_secure_pow_fractional_precision(
        base: int, bit_precision: int
    ) -> None:
        """
        Test the precision of the secure pow function

        :param base: base of the exponentiation
        :param bit_precision: Bound on absolute error of the approximation, which
        should be at most 2**-bit_precision.
        """
        plaintext_list = [xi / 10 for xi in range(-10, 11)]
        x_powed = [base ** xi for xi in plaintext_list]
        x_sec = mpc.input(list(map(secfxp, plaintext_list)), 0)
        return_value = _secure_pow_fractional(base, x_sec, bit_precision=bit_precision)
        return_value = [float(_) for _ in await mpc.output(return_value)]
        assert return_value == pytest.approx(x_powed, abs=2 ** -bit_precision)


class TestSecurePow:
    """
    Tests for exponentiation b**x
    """

    @staticmethod
    @pytest.mark.parametrize("base", (2, 3))
    @pytest.mark.parametrize("secnum", (secint, secint16))
    def test_secure_pow_type_secint(base: int, secnum: Type[SecureInteger]) -> None:
        """
        Verify type of return value.

        :param base: base of the exponentiation
        :param secnum: secure number type
        """
        plaintext_list = list(range(-10, 11))
        x_sec = mpc.input(list(secnum(_) for _ in plaintext_list), 0)
        return_value = secure_pow(base, x_sec)
        assert isinstance(return_value, list)
        assert len(return_value) == len(plaintext_list)
        assert all(isinstance(_, secnum) for _ in return_value)

    @staticmethod
    @pytest.mark.parametrize("base", (2, 3, 2.0, math.e))
    @pytest.mark.parametrize("secnum", (secfxp, secfxp16))
    def test_secure_pow_type_secfxp(
        base: float, secnum: Type[SecureFixedPoint]
    ) -> None:
        """
        Verify type of return value.

        :param base: base of the exponentiation
        :param secnum: secure number type
        """
        plaintext_list = [_ / 10 for _ in range(-10, 11)]
        x_sec = mpc.input(list(secnum(_, integral=False) for _ in plaintext_list), 0)
        return_value = secure_pow(base, x_sec)
        assert isinstance(return_value, list)
        assert len(return_value) == len(plaintext_list)
        assert all(isinstance(_, secnum) for _ in return_value)

    @staticmethod
    @pytest.mark.parametrize("base", (math.e,))
    @pytest.mark.parametrize("secnum", (secint, secint16))
    def test_secure_pow_type_exception(
        base: float, secnum: Type[SecureInteger]
    ) -> None:
        """
        Verify type of return value.

        :param base: base of the exponentiation
        :param secnum: secure number type
        """
        plaintext_list = list(range(-10, 11))
        x_sec = mpc.input(list(secnum(_) for _ in plaintext_list), 0)
        with pytest.raises(TypeError):
            secure_pow(base, x_sec)

    @staticmethod
    @pytest.mark.parametrize("base", (-3, -2, 2, 3))
    @pytest.mark.parametrize("secnum", (secint, secint16))
    async def test_secure_pow_outcome_int_secint(
        base: int, secnum: Type[SecureInteger]
    ) -> None:
        """
        Verify that the secure exponentiation returns the correct result.

        :param base: base of the exponentiation
        :param secnum: secure number type
        """
        plaintext_list = list(range(10))
        secure_list = mpc.input(list(map(secnum, plaintext_list)), 0)
        plaintext_list_powed = [base ** xi for xi in plaintext_list]
        return_value = secure_pow(base, secure_list)
        assert await mpc.output(return_value) == plaintext_list_powed

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("base", (-3, -2, 2, 3))
    @pytest.mark.parametrize("secnum", (secfxp, secfxp16))
    async def test_secure_pow_outcome_int_secfxp(
        base: int, secnum: Type[SecureFixedPoint]
    ) -> None:
        """
        Verify that the secure exponentiation returns the correct result.

        :param base: base of the exponentiation
        :param secnum: secure number type
        """
        plaintext_list = list(range(10))
        secure_list = mpc.input(
            list(secnum(_, integral=True) for _ in plaintext_list), 0
        )
        plaintext_list_powed = [base ** xi for xi in plaintext_list]
        return_value = secure_pow(base, secure_list)

        assert await mpc.output(return_value) == plaintext_list_powed

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("base", (0.5, 1.23, 2.0, 2, 3.0, math.e))
    @pytest.mark.parametrize("secnum", (secfxp, secfxp16))
    async def test_secure_pow_outcome_float(
        base: int, secnum: Type[SecureFixedPoint]
    ) -> None:
        """
        Verify that the secure exponentiation returns the correct result.

        :param base: base of the exponentiation
        :param secnum: secure number type
        """
        plaintext_list = [_ / 10 for _ in range(-80, 101)]
        secure_list = mpc.input(
            list(secnum(_, integral=False) for _ in plaintext_list), 0
        )
        plaintext_list_powed = [base ** xi for xi in plaintext_list]
        return_value = secure_pow(base, secure_list)
        bit_precision = secfxp.frac_length // 2

        assert await mpc.output(return_value) == pytest_approx_default(
            plaintext_list_powed
        )


class TestSecurePowDomains:
    """
    Tests for exponentiation 2**x where x is in various domains. Depending on
    the domain that (presumably) contains x, pre- and postprocessing may take
    place in the internals of secure_pow.
    """

    @staticmethod
    async def test_secint_negative_lower_bound_raises_error() -> None:
        """
        Output can not be of type secint if the exponent is negative. Verify
        that providing a negative lower bound on the domain of the exponent x
        raises an error.
        """
        secnum = secint
        with pytest.raises(AssertionError):
            secure_pow(
                DEFAULT_BASE,
                secnum(1),
                trunc_to_domain=False,
                lower=-1,
                upper=1,
            )

    @staticmethod
    async def test_secint_domain_too_large_raises_error() -> None:
        """
        Verify that an error is raised if the provided domain of exponents is
        too large.
        """
        with pytest.raises(AssertionError):
            secure_pow(
                DEFAULT_BASE,
                secint(1),
                trunc_to_domain=False,
                lower=0,
                upper=maximal_exponent_secint + 1,
            )

    @staticmethod
    @pytest.mark.asyncio
    async def test_secint_without_trunc_accurate() -> None:
        """
        Verify that secure_pow branch "trunc_to_domain=False" returns accurate
        results of the correct type when secint are provided. Note that we
        always end up in the subbranch "domain_length <= max_upper_bound" due
        to the non-negative domain.
        """
        upper = floor_to_odd_integer(
            maximal_exponent_secint
        )  # Ensure odd number so that center of interval is non-integral
        plaintext_list = list(
            range(upper)
        )  # Values in domain for meaningful comparison
        secure_list = mpc.input(list(map(secint, plaintext_list)), 0)
        plaintext_list_powed = [DEFAULT_BASE ** xi for xi in plaintext_list]
        return_value = secure_pow(
            DEFAULT_BASE,
            secure_list,
            trunc_to_domain=False,
            lower=0,
            upper=upper,
        )
        assert all(isinstance(_, secint) for _ in return_value)
        assert await mpc.output(return_value) == plaintext_list_powed

    @staticmethod
    @staticmethod
    @pytest.mark.asyncio
    async def test_secint_without_trunc_no_error() -> None:
        """
        Verify that secure_pow branch "trunc_to_domain=False" does not raise
        an error when exponents outside of the domain are provided as input.
        """
        upper = floor_to_odd_integer(
            maximal_exponent_secint
        )  # Ensure odd number so that center of interval is non-integral
        plaintext_list = list(
            range(2 * upper)
        )  # Include values outside of domain, shouldn't raise an exception
        secure_list = mpc.input(list(map(secint, plaintext_list)), 0)
        plaintext_list_powed = [DEFAULT_BASE ** xi for xi in plaintext_list]
        return_value = secure_pow(
            DEFAULT_BASE,
            secure_list,
            trunc_to_domain=False,
            lower=0,
            upper=upper,
        )
        assert all(
            isinstance(value, secint)
            for value, exponent in zip(return_value, plaintext_list)
            if exponent <= upper
        )
        assert await mpc.output(return_value) == plaintext_list_powed

    @staticmethod
    @pytest.mark.asyncio
    async def test_secint_domain_with_trunc_accurate() -> None:
        """
        Verify that secure_pow branch "trunc_to_domain=True" returns accurate
        results of the correct type when secint are provided.
        """
        lower = 0
        upper = floor_to_odd_integer(
            maximal_exponent_secint
        )  # Ensure odd number so that center of interval is non-integral
        plaintext_list = list(range(2 * upper))  # Include values outside of domain
        secure_list = mpc.input(list(map(secint, plaintext_list)), 0)
        plaintext_list_powed = [
            truncate_exponented_value(
                value=DEFAULT_BASE ** xi,
                lower_bound=DEFAULT_BASE ** lower,
                lower_fill=0,
                upper_bound=DEFAULT_BASE ** upper,
                upper_fill=DEFAULT_BASE ** upper,
            )
            for xi in plaintext_list
        ]
        return_value = secure_pow(
            DEFAULT_BASE,
            secure_list,
            trunc_to_domain=True,
            lower=lower,
            upper=upper,
        )
        assert all(isinstance(_, secint) for _ in return_value)
        assert await mpc.output(return_value) == plaintext_list_powed

    @staticmethod
    @pytest.mark.asyncio
    async def test_secfxp_half_domain_without_trunc_accurate() -> None:
        """
        Verify that secure_pow branch "trunc_to_domain=False, domain_length <=
        max_upper_bound" returns accurate results of the correct type when
        secfxp are provided.
        """
        lower = 0
        upper = maximal_exponent_secfxp
        plaintext_list = [
            _ / 2 for _ in range(2 * lower, 2 * upper)
        ]  # Values in domain for meaningful comparison
        secure_list = [mpc.input(secfxp(_, integral=False), 0) for _ in plaintext_list]
        plaintext_list_powed = [DEFAULT_BASE ** xi for xi in plaintext_list]
        return_value = secure_pow(
            DEFAULT_BASE,
            secure_list,
            trunc_to_domain=False,
            lower=lower,
            upper=upper,
        )
        assert all(isinstance(_, secfxp) for _ in return_value)
        assert await mpc.output(return_value) == pytest_approx_default(
            plaintext_list_powed
        )

    @staticmethod
    @pytest.mark.asyncio
    async def test_secfxp_domain_without_trunc_accurate() -> None:
        """
        Verify that secure_pow branch "trunc_to_domain=False, domain_length >
        max_upper_bound" returns accurate results of the correct type when
        secfxp are provided.
        """
        lower = minimal_exponent_secfxp
        upper = maximal_exponent_secfxp
        plaintext_list = [
            _ / 2 for _ in range(2 * lower, 2 * upper)
        ]  # Values in domain for meaningful comparison
        secure_list = [mpc.input(secfxp(_, integral=False), 0) for _ in plaintext_list]
        plaintext_list_powed = [DEFAULT_BASE ** xi for xi in plaintext_list]
        return_value = secure_pow(
            DEFAULT_BASE,
            secure_list,
            trunc_to_domain=False,
            lower=lower,
            upper=upper,
        )
        assert all(isinstance(_, secfxp) for _ in return_value)
        assert await mpc.output(return_value) == pytest_approx_default(
            plaintext_list_powed
        )

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("lower", (0, minimal_exponent_secfxp))
    async def test_secfxp_too_large_domain_without_trunc_no_error(lower: int) -> None:
        """
        Verify that secure_pow branch "trunc_to_domain=False" does not raise
        an error when exponents outside of the domain are provided as input.

        :param lower: Lower bound of tested domain.
        """
        upper = maximal_exponent_secfxp
        plaintext_list = [
            _ / 2 for _ in range(2 * lower - 10, 2 * upper + 10)
        ]  # Include values outside of domain, shouldn't raise an exception
        secure_list = [mpc.input(secfxp(_, integral=False), 0) for _ in plaintext_list]
        plaintext_list_powed = [DEFAULT_BASE ** xi for xi in plaintext_list]
        return_value = secure_pow(
            DEFAULT_BASE,
            secure_list,
            trunc_to_domain=False,
            lower=lower,
            upper=upper,
        )
        assert all(isinstance(_, secfxp) for _ in return_value)

        return_value_final = await mpc.output(return_value)
        assert all(
            value_mpc == pytest_approx_default(value_plain)
            for value_mpc, value_plain, exponent in zip(
                return_value_final, plaintext_list_powed, plaintext_list
            )
            if lower <= exponent <= upper
        )

    @staticmethod
    @pytest.mark.asyncio
    async def test_secfxp_domain_with_trunc_no_error() -> None:
        """
        Verify that secure_pow branch "trunc_to_domain=True" returns accurate
        results of the correct type when secfxp are provided.
        """
        lower = minimal_exponent_secfxp
        upper = maximal_exponent_secfxp
        plaintext_list = [
            _ / 2 for _ in range(2 * lower - 10, 2 * upper + 10)
        ]  # Include values outside of domain
        secure_list = [mpc.input(secfxp(_, integral=False), 0) for _ in plaintext_list]
        plaintext_list_powed = [
            truncate_exponented_value(
                value=DEFAULT_BASE ** xi,
                lower_bound=DEFAULT_BASE ** lower,
                lower_fill=0,
                upper_bound=DEFAULT_BASE ** upper,
                upper_fill=DEFAULT_BASE ** upper,
            )
            for xi in plaintext_list
        ]
        return_value = secure_pow(
            DEFAULT_BASE,
            secure_list,
            trunc_to_domain=True,
            lower=lower,
            upper=upper,
        )
        assert all(isinstance(_, secfxp) for _ in return_value)
        assert await mpc.output(return_value) == pytest.approx(
            plaintext_list_powed,
            abs=2 ** -(DEFAULT_BIT_PRECISION // 2),
            rel=2 ** -(DEFAULT_BIT_PRECISION // 2 - 1),
        )

    @staticmethod
    def test_not_implemented_neg_base() -> None:
        """
        Verify output of conversions for edge cases.
        """
        with pytest.raises(NotImplementedError):
            base = -2
            secure_value = mpc.input(secfxp(2, integral=False), 0)
            secure_pow(base, secure_value)

    @staticmethod
    @pytest.mark.asyncio
    async def test_buffer_decreases_domain() -> None:
        """
        Verify that bit_buffer reduces the feasible domain.
        """
        base = DEFAULT_BASE
        # Positive exponent + buffer-induced decreased domain -> reduced outcome
        secure_value = mpc.input(secfxp(50, integral=False), 0)
        no_buffer_return_value = secure_pow(
            base, secure_value, bits_buffer=0, trunc_to_domain=True
        )
        yes_buffer_return_value = secure_pow(
            base, secure_value, bits_buffer=5, trunc_to_domain=True
        )
        assert await mpc.output(no_buffer_return_value) > await mpc.output(
            yes_buffer_return_value
        )

        # Negative exponent + buffer-induced decreased domain -> truncation
        # more easily activates
        secure_value = mpc.input(secfxp(minimal_exponent_secfxp + 3, integral=False), 0)
        no_buffer_return_value = secure_pow(
            base, secure_value, bits_buffer=0, trunc_to_domain=True
        )
        yes_buffer_return_value = secure_pow(
            base, secure_value, bits_buffer=5, trunc_to_domain=True
        )
        # Sanity check: without buffer, the return value is nonzero
        assert await mpc.output(no_buffer_return_value) > 0
        # With buffer, the solution is not in the feasible domain and
        # truncation was activated
        assert await mpc.output(yes_buffer_return_value) == 0

    @staticmethod
    def test_large_buffer_raise_error() -> None:
        """
        Verify error is raised if buffer is too large (e.g. feasible domain
        empty).
        """
        with pytest.raises(AssertionError):
            base = 2
            secure_value = mpc.input(secfxp(2, integral=False), 0)
            secure_pow(base, secure_value, bits_buffer=30)
