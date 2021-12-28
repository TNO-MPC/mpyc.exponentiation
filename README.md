# TNO MPC Lab - MPyC - Secure Exponentiation

The TNO MPC lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of MPC solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed MPC functionalities to boost the development of new protocols and solutions.

The package tno.mpc.mpyc.exponentiation is part of the TNO Python Toolbox.

The package contains a generic extension to MPyC that allows you to securely compute $`a^{[x]}`$, where $`a`$ is a non-negative, plain-text base (floating point or integer) and $`x`$ is a secure number (SecFxp or SecInt).

*Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws.*  
*This implementation of cryptographic software has not been audited. Use at your own risk.*

## Documentation

Documentation of the tno.mpc.mpyc.exponentiation package can be found [here](https://docs.mpc.tno.nl/mpyc/exponentiation/1.6.2).

## Install

Easily install the tno.mpc.mpyc.exponentiation package using pip:
```console
$ python -m pip install tno.mpc.mpyc.exponentiation
```

### Note:
A significant performance improvement can be achieved by installing the GMPY2 library.
```console
$ python -m pip install 'tno.mpc.mpyc.exponentiation[gmpy]'
```

If you wish to run the tests you can use:
```console
$ python -m pip install 'tno.mpc.mpyc.exponentiation[tests]'
```

## Usage

### Minimal example

> `example.py`
> ```python
> import math
> 
> from mpyc.runtime import mpc
> 
> from tno.mpc.mpyc.exponentiation import secure_pow
> 
> 
> async def main(base=math.e, x=[-1.5, 2.3, 4.5]):
>     async with mpc:
>         stype = mpc.SecFxp()
>         sec_x = (
>             [stype(xx, integral=False) for xx in x] if isinstance(x, list) else stype(x)
>         )
>         result = secure_pow(base, sec_x)
>         revealed_result = await mpc.output(result)
>     plain_result = [base ** xx for xx in x] if isinstance(x, list) else base ** x
>     print(f"Result of secure exponentiation is {revealed_result}")
>     print(f"In the plain we would have gotten  {plain_result}")
>     diff = (
>         [abs(sec - plain) for sec, plain in zip(revealed_result, plain_result)]
>         if isinstance(x, list)
>         else abs(revealed_result - plain_result)
>     )
>     print(f"The absolute difference is {diff}")
> 
> 
> if __name__ == "__main__":
>     mpc.run(main())
> ```
