include("CMBLensing_spt3g_interface/interface.jl")

@unpack f, ds = load_sim_dataset(θpix=2, Nside=32, pol=:IP)
fQ = f[:Q]

# jl -> py auto-convert
py"type($fQ)" # MapSpectrum2D
py"type($(Map(fQ)))" # FlatSkyMap
py"type($f)" # dict
py"type($(MapSpectraTEB(f)))" # MapSpectraTEB

# infer keys but specify basis
Frame(f, FlatSkyMap)

# full flexibility
Frame(f, "TQU", MapSpectrum2D, mult=diag(ds.M[:I]))

# for transfer functions:
foo = Frame(f, mult=py"1 / get_fft_scale_fac(parent=$(Parent(f)))")
# or
MapSpectraTEB(f, mult=py"1 / get_fft_scale_fac(parent=$(Parent(f)))")
