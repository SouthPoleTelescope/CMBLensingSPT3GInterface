# CMBLensing-SPT3G Interface

## Install

The root folder of this repo is both a Julia package and a Python package. To install both, clone this repo, 

```shell
git clone https://github.com/SouthPoleTelescope/CMBLensingSPT3GInterface
```

then from the Julia package prompt run:

```
pkg> dev /path/to/CMBLensingSPT3GInterface
```

and add the folder to your `PYTHONPATH`:

```shell
export PYTHONPATH=/path/to/CMBLensingSPT3GInterface:$PYTHONPATH
```

This last step may also be done in a Python session with

```python
import sys; sys.path = ["/path/to/CMBLensingSPT3GInterface"] + sys.path
```

If you added the interface to a specific Julia project, you will also have to make sure that Python knows to use the correct project environment:

```python 
import os; os.environ["JULIA_PROJECT"] = "/path/to/project"
```

These steps must be done *before* importing the interface.


You will need a working build of [spt3g_software](https://github.com/SouthPoleTelescope/spt3g_software). Conversely, CMBLensing.jl will get installed automatically by the `dev` command above if you don't have it. 

## Usage

This package makes it so any fields passed between Julia and Python are automatically converted between the corresponding CMBLensing.jl and spt3g_software types, and all unit conversions and Fourier convention differences are automatically handled. This mapping currently includes:

| CMBLensing.jl  | spt3g_software |
|----------------|----------------|
| FlatMap        | FlatSkyMap     |
| FlatFourier    | MapSpectrum2D  |
| FlatIEBFourier | MapSpectraTEB  |
| Anything else  | dict           |

This means, for example, you can create a `FlatSkyMap` in Python and pass it to Julia, where it arrives as a `FlatMap`, or create a `FlatMap` in Julia and pass it to Python, where it arrives as a `FlatSkyMap`.

### Julia Example

```julia
julia> using CMBLensingSPT3GInterface

julia> py"""
       import numpy as np
       from spt3g.maps import FlatSkyMap, MapProjection
       from spt3g.core import G3Units
       """
       
julia> f = py"FlatSkyMap(np.random.rand(10,10), res=2*G3Units.arcmin, proj=MapProjection.ProjZEA)"
100-element FlatMap{10×10 map, 2′ pixels, fourier∂, Array{Float64,2}}:
...

julia> py"type($f)"
PyObject <class 'spt3g.maps.FlatSkyMap'>

```

### Python Example

```python
In [1]: from CMBLensingSPT3GInterface import jl

In [2]: f = jl("FlatMap(rand(10,10))"); f
Out[2]: <spt3g.maps.FlatSkyMap at 0x7f7f370d2c30>

In [3]: jl("typeof($f)") 
Out[3]: <PyCall.jlwrap FlatMap{10×10 map, 1′ pixels, fourier∂, Array{Float64,2}}>
```

### Units

CMBLensing.jl fields do not currently track their units. By default, the conversion always assumes that these fields are in μK units, and Fourier transforms are in 1/μK units. If instead the field is unitless (for the example, if the field represents a mask whose values may lie between 0 and 1), wrap the field in `unitless` before conversion, e.g.:


```python
python> jl("unitless(FlatMap(rand(10,10)))")
<spt3g.maps.FlatSkyMap at 0x7fee41d26180>
```

In the other direction, spt3g_software fields _do_ track their units (in `FlatSkyMap.units`), so no such manual wrapper like `unitless` is needed.


## Notes

### Map Orientation

`CMBLensing.plot()` defaults to `origin="upper"` which will result in maps flipped in the $y$ direction compared to plotting tools in spt3g_software, which default to `origin="lower"`. Pass `origin="lower"` to CMBLensing.plot() to make CMBLensing use the 3g software convention."