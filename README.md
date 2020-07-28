# CMBLensing-SPT3G Interface

## Install

The root folder of this repo is both a Julia package and a Python package. To install both, clone this repo, 

```shell
git clone https://github.com/SouthPoleTelescope/CMBLensingSPT3GInterface
```

then from the Julia package prompt run:

```
pkg> dev /path/to/CMBLensingSPT3GInterface
pkg> add https://github.com/marius311/PyCall.jl#pytype_mapping_prec
```

and add the folder to your `PYTHONPATH`:

```shell
export PYTHONPATH=/path/to/CMBLensingSPT3GInterface:$PYTHONPATH
```


You will need a working installation of [spt3g_software](https://github.com/SouthPoleTelescope/spt3g_software). Conversely, CMBLensing.jl will get installed automatically by the `dev` command above if you don't have it. 

## Usage

This package makes it so any fields passed between Julia and Python are automatically converted between the corresponding CMBLensing.jl and spt3g_software types, and all unit conversions and Fourier convention differences are automatically handled. This mapping currently includes:

| CMBLensing.jl  | spt3g_software |
|----------------|----------------|
| FlatMap        | FlatSkyMap     |
| FlatFourier    | MapSpectrum2D  |
| FlatIEBFourier | MapSpectraTEB  |
| Anything else  | dict           |

This means, for example, you can create a `FlatSkyMap` in Python and pass it to Julia, where it arrives as a `FlatMap`:

```julia
julia> using CMBLensingSPT3GInterface

julia> py"FlatSkyMap(np.rand(10,10))" 
100-element FlatMap{10×10 map, 1′ pixels, fourier∂, Array{Float64,2}}
...
```

or the other way around, create the map in Julia and pass it to Python:

```python
python> from CMBLensingSPT3GInterface import jl

python> jl("FlatMap(rand(10,10))")
<spt3g.maps.FlatSkyMap at 0x7fee41d26180>
```

CMBLensing.jl fields do not currently track their units. By default, the conversion always assumes that these fields are in μK units, and Fourier transforms are in 1/μK units. If instead the field is unitless (for the example, if the field represents a mask whose values may lie between 0 and 1), wrap the field in `unitless` before conversion, e.g.:


```python
python> jl("unitless(FlatMap(rand(10,10)))")
<spt3g.maps.FlatSkyMap at 0x7fee41d26180>
```

In the other direction, spt3g_software fields _do_ track their units (in `FlatSkyMap.units`), so no such manual wrapper like `unitless` is needed.


## Notes

If you get a method redefinition warning (which is harmless, but annoying), you can get rid of it by doing 

```
pkg> add https://github.com/marius311/PyCall.jl#pytype_mapping_prec
```

into the environment where you're using this package from (this was included in the install instructions above).
