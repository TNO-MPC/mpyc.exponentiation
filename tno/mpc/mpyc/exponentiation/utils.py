"""
Utility functions for exponent.py
"""
from typing import List, Type, TypeVar, Union, overload

from mpyc import gmpy as gmpy2
from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint, SecureInteger, SecureNumber

from tno.mpc.mpyc.stubs.asyncoro import mpc_coro_ignore, returnType

SecureNumberType = TypeVar("SecureNumberType", bound=SecureNumber)


@overload
def convert_to_secint(to_convert: SecureFixedPoint) -> SecureInteger:
    ...


@overload
def convert_to_secint(to_convert: List[SecureFixedPoint]) -> List[SecureInteger]:
    ...


@mpc_coro_ignore
async def convert_to_secint(
    to_convert: Union[SecureFixedPoint, List[SecureFixedPoint]]
) -> Union[SecureInteger, List[SecureInteger]]:
    """
    Converts (list of) to_convert of secure type to secint with the same modulus.
    to_convert is assumed to be a share of an integer.

    :param to_convert: (list of) secure type
    :return: converted (list of) secure integers
    """
    is_list = isinstance(to_convert, list)
    to_convert_list: List[SecureFixedPoint]
    if is_list:
        to_convert_list = to_convert[:]  # type: ignore[index]
    else:
        to_convert_list = [to_convert]  # type: ignore[list-item]
    stype = type(to_convert_list[0])
    field_modulus = stype.field.modulus
    bit_length = stype.bit_length or 0
    secint: Type[SecureInteger] = mpc.SecInt(
        l=bit_length - stype.frac_length, p=field_modulus
    )
    assert secint.field is not None
    if is_list:
        await returnType(secint, len(to_convert_list))
    else:
        await returnType(secint)
    to_convert_list = await mpc.gather(to_convert_list)
    frac_inv = gmpy2.invert(2 ** stype.frac_length, field_modulus)
    result: List[SecureInteger] = [
        secint.field(int(_.value * frac_inv % field_modulus)) for _ in to_convert_list
    ]

    if not is_list:
        return result[0]
    return result


@overload
def convert_to_secfxp(
    to_convert: SecureNumberType,
    secfxp: Type[SecureFixedPoint],
) -> SecureFixedPoint:
    ...


@overload
def convert_to_secfxp(
    to_convert: List[SecureNumberType],
    secfxp: Type[SecureFixedPoint],
) -> List[SecureFixedPoint]:
    ...


@mpc_coro_ignore
async def convert_to_secfxp(
    to_convert: Union[SecureFixedPoint, List[SecureFixedPoint]],
    secfxp: Type[SecureFixedPoint],
) -> Union[SecureFixedPoint, List[SecureFixedPoint]]:
    """
    Converts (list of) to_convert of secure type to secfxp. Secfxp is assumed to
    have more fractional bits than type(to_convert).

    :param to_convert: (list of) secure type
    :param secfxp: type to use for conversion
    :return: converted (list of) secure fixed points
    """
    is_list = isinstance(to_convert, list)
    to_convert_list: List[SecureFixedPoint]
    if is_list:
        to_convert_list = to_convert[:]  # type: ignore[index]
    else:
        to_convert_list = [to_convert]  # type: ignore[list-item]
    assert all(
        _.frac_length <= secfxp.frac_length for _ in to_convert_list
    ), "to_convert has more fractional bits than secfxp; conversion to secfxp not possible."

    if is_list:
        await returnType(secfxp, len(to_convert_list))
    else:
        await returnType(secfxp)

    frac_length = (
        to_convert_list[0].frac_length
        if isinstance(to_convert_list[0], SecureFixedPoint)
        else 0
    )

    field_modulus = to_convert_list[0].field.modulus
    field_signed = to_convert_list[0].field.is_signed
    to_convert_list = await mpc.gather(to_convert_list)
    assert secfxp.field.modulus == field_modulus or not (
        field_signed and any(_.value > field_modulus >> 1 for _ in to_convert_list)
    ), "Modulus of to_convert is different than modulus of secfxp; conversion to secfxp not (yet) possible."
    result: List[SecureFixedPoint] = [
        secfxp.field(
            (_.value - (field_modulus >> 1)) * 2 ** (secfxp.frac_length - frac_length)
            + (secfxp.field.modulus >> 1) * 2 ** (secfxp.frac_length - frac_length)
            if field_signed and _.value > field_modulus >> 1
            else _.value * 2 ** (secfxp.frac_length - frac_length)
        )
        for _ in to_convert_list
    ]
    if not is_list:
        return result[0]
    return result


@overload
def secure_ge_vec(
    value_or_list_1: SecureNumberType,
    value_or_list_2: SecureNumberType,
    strict: bool = ...,
) -> SecureNumberType:
    ...


@overload
def secure_ge_vec(
    value_or_list_1: List[SecureNumberType],
    value_or_list_2: SecureNumberType,
    strict: bool = ...,
) -> List[SecureNumberType]:
    ...


@overload
def secure_ge_vec(
    value_or_list_1: SecureNumberType,
    value_or_list_2: List[SecureNumberType],
    strict: bool = ...,
) -> List[SecureNumberType]:
    ...


@overload
def secure_ge_vec(
    value_or_list_1: List[SecureNumberType],
    value_or_list_2: List[SecureNumberType],
    strict: bool = ...,
) -> List[SecureNumberType]:
    ...


def secure_ge_vec(
    value_or_list_1: Union[List[SecureNumberType], SecureNumberType],
    value_or_list_2: Union[List[SecureNumberType], SecureNumberType],
    strict: bool = False,
) -> Union[SecureNumberType, List[SecureNumberType]]:
    """
    Securely performs $a >= b$ elementwise.

    :param value_or_list_1: first list or single value to use in comparison
    :param value_or_list_2: second list or single value to use in comparison
    :param strict: set to true to perform $>$ instead of $>=$
    :return: bitvector containing the result of the comparison
    :raises ValueError: raised when the two imput vectors have unequal ($>1$) lengths
    """
    # TODO: Fix vectorised version (mpc.prod does not handle secfxp properly).
    # result = vector_sge(a, b)

    a_is_list = True
    b_is_list = True

    if not isinstance(value_or_list_1, list):
        a_is_list = False
        value_list_1 = [value_or_list_1]
    else:
        value_list_1 = value_or_list_1
    if not isinstance(value_or_list_2, list):
        b_is_list = False
        value_list_2 = [value_or_list_2]
    else:
        value_list_2 = value_or_list_2

    if not len(value_list_1) == len(value_list_2):
        if len(value_list_1) == 1:
            value_list_1 = value_list_1 * len(value_list_2)
        elif len(value_list_2) == 1:
            value_list_2 = value_list_2 * len(value_list_1)
        else:
            raise ValueError(
                f"Cannot compare vectors of unequal length: {len(value_list_1)} != {len(value_list_2)}."
            )

    result = [
        aa >= bb if not strict else aa > bb
        for aa, bb in zip(value_list_1, value_list_2)
    ]

    if not a_is_list and not b_is_list:
        return result[0]

    return result
