"""
Module containing all logic for securely performing exponentiation using the MPyC framework.
"""

import math
from random import randint
from typing import List, Optional, Type, TypeVar, Union, overload

from mpyc import gmpy as gmpy2
from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint, SecureInteger, SecureNumber

from tno.mpc.mpyc.stubs.asyncoro import mpc_coro_ignore, returnType

from .precision_ledger import PrecisionLedger
from .utils import convert_to_secfxp, convert_to_secint, secure_ge_vec

precision_ledger = PrecisionLedger()
SecureNumberType = TypeVar("SecureNumberType", bound=SecureNumber)
SecureObjectsTV = TypeVar("SecureObjectsTV", SecureInteger, SecureFixedPoint)
SecureObjectsContainerTV = TypeVar(
    "SecureObjectsContainerTV", List[SecureInteger], List[SecureFixedPoint]
)


def minimal_exponent(
    base: float, stype: Union[Type[SecureFixedPoint], Type[SecureInteger]]
) -> int:
    """
    Returns smallest exponent x such that base**x fits within the precision of
    the field.

    :param base: base of the exponentiation
    :param stype: SecureNumberType class
    :return: smallest exponent x such that base**x fits within the precision of the given types
        field
    """
    if issubclass(stype, SecureFixedPoint):
        return math.ceil(math.log(2 ** -stype.frac_length, base))
    return 0


def maximal_exponent(
    base: float, stype: Union[Type[SecureFixedPoint], Type[SecureInteger]]
) -> int:
    """
    Returns largest exponent x such that base**x fits within the precision of
    the field.

    :param base: base of the exponentiation
    :param stype: SecureNumberType class
    :return: largest exponent x such that base**x fits within the precision of the given types
        field
    """
    assert stype.bit_length is not None and stype.field is not None
    if issubclass(stype, SecureFixedPoint):
        return math.floor(
            math.log(
                2 ** (stype.bit_length - stype.frac_length - stype.field.is_signed),
                base,
            )
        )
    return math.floor(
        math.log(
            2 ** (stype.bit_length - stype.field.is_signed),
            base,
        )
    )


def _determine_nr_terms_for_precision(base: int, bit_precision: int) -> int:
    r"""
    Compute the number of terms required for the Taylor approximation of the
    exponential function with $|exponent|<1$ to achieve a certain precision.

    :param base: Base of the exponentiation.
    :param bit_precision: Number of bits $b$ so that the absolute error of the
        Taylor approximation is at most $2^{-b}$.
    :return: Number of terms in the Taylor approximation to achieve the
        required precision.
    """
    return precision_ledger.nr_terms(base, bit_precision)


def convert_exponents_to_other_base(
    old_base: Union[int, float], new_base: float, exponents: List[SecureNumberType]
) -> List[SecureNumberType]:
    r"""
    Convert the exponent of a given exponentiation for a given new base so that the result is the same.

    More specifically, the following invariant is preserved:
    $base_{old}\^exponent == base_{new}\^return_value$

    :param old_base: Original base of the exponentiation.
    :param new_base: Desired base of the exponentiation.
    :param exponents: List of original exponents.
    :return: List of exponents that satisfy the stated invariant.
    """
    stype = type(exponents[0])
    if isinstance(exponents[0], SecureInteger):
        return mpc.scalar_mul(stype(round(math.log(old_base, new_base))), exponents)
    return mpc.scalar_mul(stype(math.log(old_base, new_base)), exponents)


@mpc_coro_ignore
async def _secure_pow_fractional(
    base: int,
    exponents: List[SecureFixedPoint],
    bit_precision: Optional[int] = None,
) -> List[SecureFixedPoint]:
    """
    Approximate exponentiation with plaintext base and secret-shared exponent
    whose modulus is bounded by 1.

    The exponentiation is performed by means of a Taylor approximation. The
    exponent's modulus is assumed to be bounded by 1 so that the error of the
    approximation can be quantified.

    :param base: Base of the exponentiation.
    :param exponents: List of secret-shared exponents.
    :param bit_precision: Bound on absolute error of the approximation, which
        should be at most 2**-bit_precision.
    :return: Secret-shared value of the exponentiation for every exponent
        provided.
    """
    exponents_length = len(exponents)

    # Remove None, which are introduced if the original exponents were
    # integral and therefore have no 'fractional' part.
    not_none_list = [_ for _, exponent in enumerate(exponents) if exponent is not None]
    none_list = [_ for _, exponent in enumerate(exponents) if exponent is None]
    for _ in none_list[::-1]:
        exponents.pop(_)

    stype = type(exponents[0])
    await returnType(stype, exponents_length)

    # Compute individual terms for Taylor approximation
    if exponents_length == 0:
        return []
    if bit_precision is None:
        bit_precision = exponents[0].frac_length // 2

    starter = [stype(1) for _ in range(len(exponents))]
    fixed_ln = mpc.scalar_mul(stype(math.log(base, math.e)), exponents)
    squares_z = [starter, fixed_ln]  # first two terms of sum (0 and 1)
    max_sum_index = _determine_nr_terms_for_precision(base, bit_precision)
    for _ in range(
        2, max_sum_index + 1
    ):  # final max_sum_index - 1 terms (total max_sum_index + 1 terms)
        squares_z.append(mpc.schur_prod(fixed_ln, squares_z[-1]))  # (z * ln(base)) ** i

    # Combine terms into final approximation
    result = [stype(1)] * exponents_length
    for i, idx in enumerate(not_none_list):
        result[idx] = stype(0)
        for j, squares_z_element in enumerate(squares_z):
            result[idx] += 1 / math.factorial(j) * squares_z_element[i]
    return result


@mpc_coro_ignore
async def _secure_pow_int(
    base: int,
    exponents: SecureObjectsContainerTV,
    computers: Optional[List[int]] = None,
) -> SecureObjectsContainerTV:
    """
    Exact exponentiation with plaintext integral base and secret-shared
    integral exponent. Both are assumed to be positive.

    The exponentiation is performed by means of an additive masking of the
    exponent. The exponentiation of the masked exponent and the exponentiation
    of the individual masks can be performed in the clear, yielding the result
    after a secure aggregation.

    :param base: Base of the exponentiation.
    :param exponents: List of secret-shared exponents.
    :param computers: Identifiers of the (at least t+1) parties that generate
        additive masks.
    :return: Secret-shared value of the exponentiation for every exponent
        provided.
    """
    assert isinstance(base, int), "Base should be integer."
    assert all(
        isinstance(exponent, SecureInteger)
        or (isinstance(exponent, SecureFixedPoint) and exponent.integral)
        for exponent in exponents
    ), "Exponent should be integer (exponent.integral should be True)."

    stype = type(exponents[0])
    number_of_exponents = len(exponents)
    await returnType(stype, number_of_exponents)

    exponents_int: List[SecureInteger]
    if isinstance(exponents[0], SecureInteger):
        exponents_int = exponents
    else:
        exponents_int = convert_to_secint(exponents)
    ttype = type(exponents_int[0])

    assert stype.field is not None and stype.bit_length is not None
    stype_modulus = stype.field.modulus  # equal to ttype.field.modulus
    threshold = mpc.threshold
    stype_bit_length = stype.bit_length

    if computers is None:
        computers = list(range(threshold + 1))
    number_of_computers = len(computers)

    assert set(computers).issubset(
        set(range(len(mpc.parties)))
    ), f"Received invalid set of computing parties: {computers}."
    assert (
        number_of_computers >= threshold + 1
    ), f"At least {threshold + 1} parties required that generate randomness in secure exponentiation, received {number_of_computers}."
    assert stype_modulus // number_of_computers > 2 ** (
        stype_bit_length + mpc.options.sec_param
    ), (
        "Modulus of exponent is too small to satisfy security condition: 2**(l + k) < p // len(computers). Here, "
        "l is the bit length of the exponent ({l}), k is the security parameter ({k}), p is the modulus of the "
        "exponent ({p}) and computers is the set of parties that generate randomness (len={c}). This issue can be "
        "solved by manually increasing the bit length of the modulus.\n"
        "Suggestion: instantiate type(exponent) by explicitly passing the modulus (parameter p), where "
        "p = finfields.find_prime_root(l + {t0}, n). Here, finfields is imported from mpyc, l{t1} and n are as "
        "requested by instantiating type(exponent), and c is the bit length of the number of computers (default: "
        "mpc.options.threshold + 1). For the current exponent type, {stype}, the other default values are "
        "l={ldef}{fdef}, and n={ndef}.".format(
            **{
                "l": stype.bit_length,
                "k": mpc.options.sec_param,
                "p": stype.field.modulus,
                "c": number_of_computers,
                "t0": "k + c + 2"
                if isinstance(stype, SecureInteger)
                else "max(f, k+1) + c + 1",
                "t1": "" if isinstance(stype, SecureInteger) else ", f",
                "stype": stype.__name__,
                "ldef": mpc.options.bit_length,
                "fdef": "" if isinstance(stype, SecureInteger) else ", f=l//2",
                "ndef": "2",
            }
        )
    )

    # 1. Generate random numbers ri, sigma bits larger than x
    if mpc.pid in computers:
        upper_bound_r = stype_modulus // number_of_computers
        lower_bound_r = math.ceil(2 ** stype_bit_length / number_of_computers)
        r_i = [
            randint(lower_bound_r, upper_bound_r) for _ in range(number_of_exponents)
        ]

    # 2. For party in parties: secret sharing of randomness
    randomness = [
        mpc.input([ttype(_) for _ in r_i], party_i)
        if mpc.pid == party_i
        else mpc.input([ttype(None) for _ in range(number_of_exponents)], party_i)
        for party_i in computers
    ]

    # 3. Calculate [y]=[x]+sum([rm])
    y = [  # pylint: disable=C0103 # y has mathematical meaning in this context
        await mpc.output(
            mpc.sum([-exponent] + [randomness[j][i] for j in range(len(randomness))])
        )
        for i, exponent in enumerate(exponents_int)
    ]
    # Correct for sign bit (in case that a signed SecureNumber was given)
    y = [  # pylint: disable=C0103 # y has mathematical meaning in this context
        _ if _ >= 0 else _ + stype_modulus for _ in y
    ]
    mod_pow_y = [gmpy2.powmod(base, _, stype_modulus) for _ in y]
    mod_pow_y_inv = [
        gmpy2.invert(mod_pow_yy, stype_modulus) for mod_pow_yy in mod_pow_y
    ]

    # 4. Compute mod_pow_ri=base^ri % p
    if mpc.pid in computers:
        mod_pow_ri = [gmpy2.powmod(base, _, stype_modulus) for _ in r_i]

    # 5. For party in parties: secret sharing of mod_pow_ri
    mod_pow_ri_vec = [
        mpc.input([ttype(int(mod_pow_rii)) for mod_pow_rii in mod_pow_ri], party_i)
        if mpc.pid == party_i
        else mpc.input([ttype(None) for _ in range(number_of_exponents)], party_i)
        for party_i in computers
    ]

    # 6. Determine [mod_pow_r]=prod([mod_pow_ri])
    mod_pow_r = [
        mpc.prod([mod_pow_ri_vec[j][i] for j in range(len(mod_pow_ri_vec))])
        for i in range(number_of_exponents)
    ]

    # 7. Public integer division [mod_pow_r]/(base^y)^-1 % p
    result = [
        int(mod_pow_yi_inv) * mod_pow_ri
        for (mod_pow_yi_inv, mod_pow_ri) in zip(mod_pow_y_inv, mod_pow_r)
    ]
    if isinstance(exponents[0], SecureInteger):
        return result
    return convert_to_secfxp(result, stype)


@overload
def _basic_secure_pow(
    base: float,
    exponents: SecureInteger,
    bit_precision: Optional[int] = ...,
    computers: Optional[List[int]] = ...,
) -> SecureInteger:
    ...


@overload
def _basic_secure_pow(
    base: float,
    exponents: SecureFixedPoint,
    bit_precision: Optional[int] = ...,
    computers: Optional[List[int]] = ...,
) -> SecureFixedPoint:
    ...


@overload
def _basic_secure_pow(
    base: float,
    exponents: SecureObjectsContainerTV,
    bit_precision: Optional[int] = ...,
    computers: Optional[List[int]] = ...,
) -> SecureObjectsContainerTV:
    ...


def _basic_secure_pow(
    base: float,
    exponents: Union[SecureInteger, SecureFixedPoint, SecureObjectsContainerTV],
    bit_precision: Optional[int] = None,
    computers: Optional[List[int]] = None,
) -> Union[SecureInteger, SecureFixedPoint, SecureObjectsContainerTV]:
    r"""
    Secure exponentiation.

    Computes approximate secret-sharing of base**x for all given
    secret-shared x in exponents. Does not check if the encoding of base**x
    can be represented in the finite field, e.g. assumes that all exponents
    are in [0, MAX_UPPER_BOUND], so sensitive to overflow.

    :param base: Base of the exponentiation.
    :param exponents: Exponents of the base.
    :param bit_precision: Bound on relative error of approximation, set to
        half of fractional bits if None.
    :param computers: Identifiers of the (at least $t+1$) parties that generate
        additive masks.
    :return: Secret-shared value of the exponentiation for every exponent
        provided.
    :raises NotImplementedError: if any of the exponents is not a secure integer
        or secure fixed point.
    """
    # exponent is (list of) SecFxp, SecInt, assumed positive
    is_list = isinstance(exponents, list)
    exponents_list: SecureObjectsContainerTV
    if isinstance(exponents, list):
        exponents_list = list(exponents)
    else:
        exponents_list = [exponents]  # type: ignore[list-item]

    assert base > 0, "Base should be positive."
    if not isinstance(base, int):
        assert isinstance(
            exponents_list[0], SecureFixedPoint
        ), "Exponent should be SecureFixedPoint when base is not integer."
        new_base = 2
        exponents_list = convert_exponents_to_other_base(
            old_base=base,
            new_base=new_base,
            exponents=exponents_list,
        )
        base = new_base

    integral_part_exponents = exponents_list
    fractional_part_exponents: List[Optional[SecureFixedPoint]] = [None] * len(
        exponents_list
    )

    for _, exponent in enumerate(exponents_list):
        if isinstance(exponent, SecureInteger) or (
            isinstance(exponent, SecureFixedPoint) and exponent.integral
        ):
            continue
        elif isinstance(exponent, SecureFixedPoint) and not exponent.integral:
            # Determine the integer and floating point part of the exponent
            stype = type(exponent)
            integral_part_exponent = (
                mpc.trunc(exponent, f=stype.frac_length) * 2 ** stype.frac_length
            )
            integral_part_exponent.integral = True
            integral_part_exponents[_] = integral_part_exponent
            fractional_part_exponents[_] = exponent - integral_part_exponent
        else:
            raise NotImplementedError(
                f"No clue what to do with of type {type(exponent)}."
            )

    # Calculate base to power of integral_part_exponent
    pow_integral = _secure_pow_int(base, integral_part_exponents, computers)

    fractionals_exist = not all(_ is None for _ in fractional_part_exponents)
    result: SecureObjectsContainerTV
    if fractionals_exist:
        # Calculate base to power of fractional_part_exponent
        pow_fractional = _secure_pow_fractional(
            base, fractional_part_exponents, bit_precision
        )
        result = mpc.schur_prod(pow_integral, pow_fractional)
    else:
        result = pow_integral

    if not is_list:
        return result[0]
    return result


@overload
def secure_pow(
    base: float,
    exponents: SecureInteger,
    trunc_to_domain: bool = ...,
    lower: Optional[float] = ...,
    upper: Optional[float] = ...,
    bit_precision: Optional[int] = ...,
) -> SecureInteger:
    ...


@overload
def secure_pow(
    base: float,
    exponents: SecureFixedPoint,
    trunc_to_domain: bool = ...,
    lower: Optional[float] = ...,
    upper: Optional[float] = ...,
    bit_precision: Optional[int] = ...,
) -> SecureFixedPoint:
    ...


@overload
def secure_pow(
    base: float,
    exponents: SecureObjectsContainerTV,
    trunc_to_domain: bool = ...,
    lower: Optional[float] = ...,
    upper: Optional[float] = ...,
    bit_precision: Optional[int] = ...,
) -> SecureObjectsContainerTV:
    ...


@mpc_coro_ignore
async def secure_pow(
    base: float,
    exponents: Union[SecureInteger, SecureFixedPoint, SecureObjectsContainerTV],
    trunc_to_domain: bool = False,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    bit_precision: Optional[int] = None,
) -> Union[SecureInteger, SecureFixedPoint, SecureObjectsContainerTV]:
    r"""
    Secure exponentiation.

    Computes approximate secret-sharing of base**x for all given
    secret-shared x in [lower, upper]. May additionally enforce that x is
    truncated to fit in [lower, upper].

    The exponentiation is approximate since it splits the computation into an
    exact integral computation and a approximate non-integral computation. The
    approximated part leverages Taylor series.

    If exponents are truncated to the feasible domain, for a exponent $x$, the
    function returns
    $base^{upper}$ if $x > upper$,
    $base^x$ if $x\in [lower, upper]$, and
    $0$ if $x < lower$.

    :param base: Base of the exponentiation.
    :param exponents: Exponents of the base.
    :param lower: Lower bound of exponents range, maximized for
        type(exponents[0]) if None.
    :param upper: Upper bound of exponents range, maximized for
        type(exponents[0]) if None.
    :param trunc_to_domain: Truncates exponents so that they fall in the given
        range at cost of additional resources. Additionally ensures that
        base**[lower] evaluates to [0].
    :param bit_precision: Bound on relative error of approximation, set to
        half of fractional bits if None.
    :return: Secret-shared value of the exponentiation for every exponent
        provided.
    :raise NotImplementedError: Raised when size of domain is greater than twice
        the size of the maximum upper bound.
    """
    exponent_is_list = isinstance(exponents, list)
    exponents_list: SecureObjectsContainerTV
    if isinstance(exponents, list):
        exponents_list = list(exponents)
    else:
        exponents_list = [exponents]  # type: ignore[list-item]
    number_of_exponents = len(exponents_list)
    stype = type(exponents_list[0])
    if exponent_is_list:
        await returnType(stype, len(exponents_list))
    else:
        await returnType(stype)

    max_lower_bound = minimal_exponent(base, stype)
    max_upper_bound = maximal_exponent(base, stype)

    if upper is None:
        upper = max_upper_bound

    if lower is None:
        # Prevent NotImplementedError
        lower_corrected = max_lower_bound
        lower_corrected = (
            lower_corrected if lower_corrected >= -max_upper_bound else -max_upper_bound
        )
        lower = lower_corrected

    assert lower < upper, "Upper bound should be larger than lower bound."
    assert (
        lower >= max_lower_bound
    ), f"Lower bound exceeds achievable precision, should be larger than MAX_LOWER_BOUND = {max_lower_bound}."
    assert (
        upper <= max_upper_bound
    ), f"Upper bound exceeds achievable precision, should be smaller than MAX_UPPER_BOUND = {max_upper_bound}."
    domain_length = upper - lower
    if domain_length > 2 * max_upper_bound:
        raise NotImplementedError(
            f"Implementation does not yet support lower bound smaller than -MAX_UPPER_BOUND = {-max_upper_bound}."
        )

    pow_exponents: SecureObjectsContainerTV
    if not trunc_to_domain:
        if domain_length <= max_upper_bound:
            # No secure comparisons
            shift = 0 if lower >= 0 else math.ceil(-lower)
            positive_exponents = mpc.vector_add(
                exponents_list, [stype(shift)] * number_of_exponents
            )

            pow_positive_exponents = _basic_secure_pow(
                base, positive_exponents, bit_precision
            )

            if shift:
                pow_exponents = mpc.scalar_mul(
                    stype(base ** -shift), pow_positive_exponents
                )
            else:
                pow_exponents = pow_positive_exponents
        else:
            # One secure comparison
            pow_exponents = _secure_pow_signed_domain_without_trunc(
                base, exponents_list, lower, bit_precision
            )
    else:
        # Two secure comparisons
        if lower < 0:
            # Domain is internally centered around zero
            pow_exponents = _secure_pow_signed_domain_with_trunc(
                base, exponents_list, lower, upper, bit_precision
            )
        else:
            # No centering needed, this part can be optimized (see issue)
            pow_exponents = _secure_pow_unsigned_domain_with_trunc(
                base, exponents_list, lower, upper, bit_precision
            )

    if not exponent_is_list:
        return pow_exponents[0]

    return pow_exponents


@mpc_coro_ignore
async def _secure_pow_signed_domain_without_trunc(
    base: float,
    exponents: List[SecureFixedPoint],
    lower: float,
    bit_precision: Optional[int] = None,
) -> List[SecureFixedPoint]:
    """
    Secure exponentiation that typically doubles the feasible range of
    exponents.

    Applies a trick to enlarge the feasible range for exponents from
    [0, MAX_UPPER_BOUND] to [MAX_LOWER_BOUND, MAX_UPPER_BOUND]. Typically,
    MAX_LOWER_BOUND = -MAX_UPPER_BOUND. Exponents are not verified to be in
    the feasible range, so sensitive to overflows.

    This modification of the vanilla secure exponentiation requires one
    additional secure comparison.

    :param base: Base of the exponentiation.
    :param exponents: Exponents of the base.
    :param lower: Lower bound of exponents range.
    :param bit_precision: Bound on relative error of approximation, set to
        half of fractional bits if None.
    :return: Secret-shared value of the exponentiation for every exponent
        provided.
    """
    stype = type(exponents[0])
    await returnType(stype, len(exponents))

    less_than_zero = secure_ge_vec(stype(0), exponents, strict=True)

    # positive_exponent = exponent     if  0 <= exponent
    #                   = exponent + L if  L <= exponent < 0
    positive_exponents = mpc.vector_add(
        exponents, mpc.scalar_mul(stype(-lower), less_than_zero)
    )

    # base ** exponent
    pow_positive_exponents: List[SecureFixedPoint] = _basic_secure_pow(
        base, positive_exponents, bit_precision
    )

    # pow_exponents = base ** capped_pow_exponent       if      exponent > 0
    #               = base ** (capped_pow_exponent - L) if  L < exponent < 0
    pow_exponents = mpc.vector_sub(
        pow_positive_exponents,
        mpc.schur_prod(
            less_than_zero,
            mpc.vector_sub(
                pow_positive_exponents,
                mpc.scalar_mul(stype(base ** lower), pow_positive_exponents),
            ),
        ),
    )
    return pow_exponents


@mpc_coro_ignore
async def _secure_pow_unsigned_domain_with_trunc(
    base: float,
    exponents: List[SecureFixedPoint],
    lower: float,
    upper: float,
    bit_precision: Optional[int] = None,
) -> List[SecureFixedPoint]:
    r"""
    Secure exponentiation that truncates exponents to be in the feasible
    range. This range should be part of the positive reals.

    Exponents are truncated to make sure they are in the feasible range. As a
    consequence, for a exponent $x$ the function returns
    $base^{upper}$ if $x > upper$,
    $base^x$ if $x\in [lower, upper]$, and
    $0$ if $x < lower$.

    This modification of the vanilla secure exponentiation requires two
    additional secure comparisons.

    :param base: Base of the exponentiation.
    :param exponents: Exponents of the base.
    :param lower: Lower bound of exponents range, should be non-negative.
    :param upper: Upper bound of exponents range.
    :param bit_precision: Bound on relative error of approximation, set to
        half of fractional bits if None.
    :return: Secret-shared value of the exponentiation for every exponent
        provided.
    """
    if not lower >= 0:
        raise NotImplementedError(
            f"This branch has only been implemented for lower >= 0, but received lower={lower}."
        )
    stype = type(exponents[0])
    number_of_exponents = len(exponents)
    await returnType(stype, number_of_exponents)

    gtr_than_lower = secure_ge_vec(exponents, stype(lower), strict=False)
    gtr_than_upper = secure_ge_vec(exponents, stype(upper), strict=True)
    is_in_domain = mpc.vector_sub(gtr_than_lower, gtr_than_upper)

    # base ** exponent
    pow_exponents = _basic_secure_pow(base, exponents, bit_precision)

    #               = base ** U        if      exponent >  U
    # pow_exponents = base ** exponent if L <= exponent <= U
    #               = 0                if      exponent <  L
    pow_exponents = mpc.vector_add(
        mpc.schur_prod(is_in_domain, pow_exponents),
        mpc.scalar_mul(stype(base ** upper), gtr_than_upper),
    )
    return pow_exponents


@mpc_coro_ignore
async def _secure_pow_signed_domain_with_trunc(
    base: float,
    exponents: List[SecureFixedPoint],
    lower: float,
    upper: float,
    bit_precision: Optional[int] = None,
) -> List[SecureFixedPoint]:
    r"""
    Secure exponentiation that typically doubles the feasible range of
    exponents and truncates exponents to be in the feasible range.

    Applies a trick to enlarge the feasible range for exponents from
    [0, MAX_UPPER_BOUND] to [MAX_LOWER_BOUND, MAX_UPPER_BOUND]. Typically,
    MAX_LOWER_BOUND = -MAX_UPPER_BOUND.

    Exponents are truncated to make sure they are in the feasible range. As a
    consequence, for a exponent $x$ the function returns
    $base^{upper}$ if $x > upper$,
    $base^x$ if $x\in [lower, upper]$, and
    $0$ if $x < lower$.

    This modification of the vanilla secure exponentiation requires two
    additional secure comparisons.

    :param base: Base of the exponentiation.
    :param exponents: Exponents of the base.
    :param lower: Lower bound of exponents range.
    :param upper: Upper bound of exponents range.
    :param bit_precision: Bound on relative error of approximation, set to
        half of fractional bits if None.
    :return: Secret-shared value of the exponentiation for every exponent
        provided.
    """
    stype = type(exponents[0])
    number_of_exponents = len(exponents)
    await returnType(stype, number_of_exponents)
    domain_length = upper - lower

    # If shift is not integer, then accuracy of integer SecureNumbers may
    # be affected
    shift = -lower - domain_length / 2
    shift = int(shift) if shift.is_integer() else shift
    centered_exponents = mpc.vector_add(exponents, [stype(shift)] * number_of_exponents)
    shifted_domain_limit = upper + shift

    shifted_gtr_than_zero = secure_ge_vec(centered_exponents, stype(0), strict=False)
    sgn_centered_exponents = mpc.vector_sub(
        mpc.scalar_mul(stype(2), shifted_gtr_than_zero),
        [stype(1)] * number_of_exponents,
    )

    #                 = U            if  U <  exponent
    # capped_exponent = exponent     if  0 <= exponent <= U
    #                 = exponent + U if -U <= exponent <  0
    #                 = 0            if       exponent < -U
    abs_centered_exponents = mpc.schur_prod(sgn_centered_exponents, centered_exponents)
    is_out_of_domain = secure_ge_vec(
        abs_centered_exponents, stype(shifted_domain_limit), strict=True
    )
    gtr_than_shifted_upper = mpc.schur_prod(shifted_gtr_than_zero, is_out_of_domain)

    is_in_domain_and_neg = mpc.vector_add(
        mpc.vector_sub(
            mpc.vector_sub([stype(1)] * number_of_exponents, shifted_gtr_than_zero),
            is_out_of_domain,
        ),
        gtr_than_shifted_upper,
    )

    capped_centered_exponents = mpc.vector_sub(
        mpc.vector_add(
            centered_exponents,
            mpc.scalar_mul(
                stype(shifted_domain_limit),
                mpc.vector_add(is_in_domain_and_neg, gtr_than_shifted_upper),
            ),
        ),
        mpc.schur_prod(is_out_of_domain, centered_exponents),
    )

    # base ** exponent
    capped_pow_centered_exponents = _basic_secure_pow(
        base, capped_centered_exponents, bit_precision
    )
    capped_pow_exponents = mpc.scalar_mul(
        stype(base ** -shift), capped_pow_centered_exponents
    )

    #               = base ** capped_pow_exponent       if      exponent >= 0
    # pow_exponents = base ** (capped_pow_exponent - L) if L <= exponent <  0
    #               = 0                                 if      exponent <  L
    pow_exponents = mpc.vector_add(
        mpc.schur_prod(shifted_gtr_than_zero, capped_pow_exponents),
        mpc.scalar_mul(
            stype(base ** -shifted_domain_limit),
            mpc.schur_prod(
                is_in_domain_and_neg,
                capped_pow_exponents,
            ),
        ),
    )
    return pow_exponents
