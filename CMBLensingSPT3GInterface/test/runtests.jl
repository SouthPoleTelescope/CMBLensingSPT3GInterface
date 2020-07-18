using CMBLensing
using CMBLensingSPT3GInterface
using PyCall
using Test

py"""
from spt3g.maps import FlatSkyMap
from spt3g.mapspectra.map_spectrum_classes import MapSpectrum2D
from spt3g.lensing.map_spec_utils import MapSpectraTEB
"""

@testset "Conversions (Nside=$Nside)" for Nside=[32,33]

    @unpack f = load_sim_dataset(θpix=2, Nside=32, T=Float64, pol=:IP)

    # CMBLensing fields end up the right type on the Python side
    @test py"isinstance($(    Map(f[:I])), FlatSkyMap)"
    @test py"isinstance($(Fourier(f[:I])), MapSpectrum2D)"
    
    # the field is unchanged after a roundtrip jl -> py -> jl
    @test py"$(    Map(f[:I]))" ≈ Map(f[:I])
    @test py"$(Fourier(f[:I]))" ≈ Fourier(f[:I])
    
    # FlatIEBFourier not auto-converted... should it be? 
    @test py"isinstance($(MapSpectraTEB(f)), MapSpectraTEB)"
    @test py"$(IEBFourier(f))" ≈ f

end