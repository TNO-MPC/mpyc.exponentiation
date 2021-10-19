"""
Module for storing and computing the required number of terms in Taylor
approximations of the exponentiation function in order to bound the absolute
error.
"""

import math
from dataclasses import dataclass
from typing import Dict


class PrecisionNrTermsException(Exception):
    """
    Required precision was not achieved with given maximum amount of terms.
    """


@dataclass(frozen=True, eq=True)
class PrecisionLedgerIndex:
    """Multi-key index for entries in the PrecisionLedger."""

    base: int
    bit_precision: int
    nr_terms_cap: int


@dataclass
class PrecisionLedgerResult:
    """Stores computational results for the PrecisionLedger."""

    nr_terms: int


class PrecisionLedger:
    """
    Object for computing and storing the required number of terms in the
    Taylor approximation of an exponentiation to achieve a certain precision.
    """

    def __init__(self) -> None:
        """Initialize instance of PrecisionLedger."""
        self._ledger: Dict[PrecisionLedgerIndex, PrecisionLedgerResult] = {}

    def nr_terms(self, base: int, bit_precision: int, nr_terms_cap: int = 15) -> int:
        """
        Returns the number of terms in the Taylor approximation of an
        exponentiation to achieve a certain precision.

        More precisely, returns the number n required to approximate base**z
        by the finite sum of ln(base)**i / i! for i from 0 to n where the
        absolute error is bounded.

        :param base: Base of the exponentiation.
        :param bit_precision: Require an 2 ** -bit_precision upper bound on
            the absolute error of the approximation for all exponents with
            modulus at most one.
        :param nr_terms_cap: Upper bound on the number of terms that is
            returned.
        :return: Minimum number of terms in the Taylor approximation to
            satisfy accuracy constraints.
        """

        index = PrecisionLedgerIndex(
            base=base, bit_precision=bit_precision, nr_terms_cap=nr_terms_cap
        )
        if index not in self._ledger.keys():
            self._ledger[index] = PrecisionLedgerResult(
                nr_terms=calculate_max_sum_index(base, bit_precision, nr_terms_cap)
            )
        return self._ledger[index].nr_terms


def calculate_max_sum_index(base: int, bit_precision: int, nr_terms_cap: int) -> int:
    r"""
    Compute the number of terms of the Taylor approximation of base ** z so
    that the absolute error is at most 2 ** -self.bit_precision.

    Guarantees hold for $|z|<1$.

    :param base: Base of the exponentiation.
    :param nr_terms_cap: Upper bound on the number of terms that is
        returned.
    :return: Minimal number of terms in the Taylor approximation to
        achieve the desired precision.
    :raise PrecisionNrTermsException: Raised when required precision cannot
        be reached
    """
    assert isinstance(base, int), "base should be integer."
    assert base > 0, "base should be positive."
    assert bit_precision >= 0, "bit_precision should be non-negative."
    assert nr_terms_cap > 0, "nr_terms_cap should be positive."

    for max_sum_index in range(0, nr_terms_cap + 1):
        if max_sum_index == 0:
            error: float = base - 1
        else:
            error -= math.log(base) ** max_sum_index / math.factorial(max_sum_index)
        if error <= 2 ** -(bit_precision + 1):
            break
    else:
        raise PrecisionNrTermsException(
            f"Required precision was not achieved with at most {nr_terms_cap} terms."
        )

    return max_sum_index
