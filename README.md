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

## Notes

If you get a method redefinition warning (which is harmless, but annoying), you can get rid of it by doing 

```
pkg> add https://github.com/marius311/PyCall.jl#pytype_mapping_prec
```

into the environment where you're using this package from (this was included in the install instructions above).