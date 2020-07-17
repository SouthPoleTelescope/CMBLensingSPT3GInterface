using Pkg; Pkg.activate("CMBLensingSPT3GInterface")
using CMBLensing, CMBLensingSPT3GInterface
using PyCall

@unpack f, ds = load_sim_dataset(θpix=2, Nside=32, pol=:IP)

# jl -> py

py"type($(f[:Q]))" # MapSpectrum2D
py"type($(Map(fQ)))" # FlatSkyMap
py"type($f)" # dict
py"type($(MapSpectraTEB(f)))" # MapSpectraTEB

# user can specify keys
Frame(f, "TQU")

# for transfer functions:
py"""
from spt3g.mapspectra.basicmaputils import get_fft_scale_fac
"""
Frame(f / py"get_fft_scale_fac(parent=$(Parent(f)))")
# or
MapSpectraTEB(f / py"get_fft_scale_fac(parent=$(Parent(f)))")



# py -> jl

py"$(f[:I])" # FlatFourier
py"$(Map(f[:I]))" # FlatMap

py"$f" #FlatIEBFourier
py"$(MapSpectraTEB(f))" #FlatIEBFourier
py"$(Map(f))" #FlatIEBMap


# for transfer functions
py"$(MapSpectraTEB(f))" * py"get_fft_scale_fac(parent=$(Parent(f)))"
# or
py"$f" * py"get_fft_scale_fac(parent=$(Parent(f)))"
