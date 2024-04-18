module ElectronSpecificHeat

using PhysicalConstants
using Unitful
using GLMakie

# Constants
const k_B = ustrip(PhysicalConstants.CODATA2018.k_B)
const ħ = ustrip(PhysicalConstants.CODATA2018.ħ)
const m_e = ustrip(PhysicalConstants.CODATA2018.m_e)

# Converts values on the x-axis from the figure on Kittel 136 
# to energy in Joules
ϵ_from_book = x -> x * k_B * 10e3

# Energy range in Joules
ϵ_min = 0
ϵ_max = ϵ_from_book(9)

# Temperature range in Kelvin
T_min = 500
T_max = 10e4

# Fermi-Dirac distribution
function fermi_dirac(ϵ, T, μ)
  return 1 / (exp((ϵ - μ) / (k_B * T)) + 1)
end

function ϵ_F(N, L)
  return ħ^2 * (3 * π^2 * N / L^3)^(2 / 3) / (2 * m_e)
end

# Plots the Fermi-Dirac distribution as a function of temperature and energy
function plot_fermi_dirac(μ)
  ϵs = range(ϵ_min, ϵ_max, length=400)
  Ts = range(T_min, T_max, length=400)
  f(ϵ, T) = fermi_dirac(ϵ, T, μ)
  fs = [f(ϵ, T) for ϵ in ϵs, T in Ts]

  fig = Figure(size=(800, 600))
  axis = Axis3(
    fig[1, 1],
    xlabel="Energy (J)",
    ylabel="Temperature (K)",
    zlabel="f(ϵ, T)"
  )
  surface!(ϵs, Ts, fs)
  return fig
end

# Plots the Fermi-Dirac distribution using the condition T_F = ϵ_F / k_B = 5e4,
# as on Kittel 136
function plot_fermi_dirac_textbook()
  μ = ϵ_F(5e4 * k_B, 1)
  return plot_fermi_dirac(μ)
end


export k_B, ħ, m_e, ϵ_from_book, ϵ_min, ϵ_max, T_min, T_max, fermi_dirac, ϵ_F, plot_fermi_dirac, plot_fermi_dirac_textbook

end # module ElectronSpecificHeat
