module ElectronSpecificHeat

using PhysicalConstants
using Unitful
using GLMakie
using ForwardDiff
using QuadGK

# Constants
const k_B = ustrip(PhysicalConstants.CODATA2018.k_B)
const ħ = ustrip(PhysicalConstants.CODATA2018.ħ)
const m_e = ustrip(PhysicalConstants.CODATA2018.m_e)

# Converts values on the x-axis from the figure on Kittel 136 
# to energy in Joules
ϵ_from_book = x -> x * k_B * 10e3

# Side length of containing cube in meters
L = 1

# Fermi-Dirac distribution
function f(ϵ, T, μ)
  return 1 / (exp((ϵ - μ) / (k_B * T) + 1) + 1)
end

# Fermi energy as a function of number of electrons and side length of containing cube
function ϵ_F(N, L)
  return (ħ^2 / (2 * m_e)) * (3 * π^2 * N / L^3)^(2 / 3)
end

# Plots the Fermi-Dirac distribution as a function of temperature and energy
function plot_fermi_dirac(μ)
  ϵ_min = 0
  ϵ_max = ϵ_from_book(9)
  T_min = 0
  T_max = 2.5e5
  ϵs = range(ϵ_min, ϵ_max, length=400)
  Ts = range(T_min, T_max, length=400)
  fermi_dirac(ϵ, T) = f(ϵ, T, μ)
  fs = [fermi_dirac(ϵ, T) for ϵ in ϵs, T in Ts]

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
# as on Kittel 136, with the assuption that μ = ϵ_F
function plot_fermi_dirac_textbook()
  μ = 5e4 * k_B
  return plot_fermi_dirac(μ)
end

# Implementation of the derivative of the Fermi-Dirac distribution with respect to temperature 
# using automatic differentiation
function df_dT(ϵ, T, μ)
  return ForwardDiff.derivative(x -> f(ϵ, x, μ), T)
end

# function df_dT(ϵ, T, μ)
#   return ((ϵ - μ) / (k_B * T^2)) * exp((ϵ - μ) / (k_B * T)) / (exp((ϵ - μ) / (k_B * T)) + 1)^2
# end

function plot_df_dT(μ)
  ϵ_min = 0
  ϵ_max = ϵ_from_book(9)
  T_min = 500
  T_max = 2.5e5
  ϵs = range(ϵ_min, ϵ_max, length=400)
  Ts = range(T_min, T_max, length=400)
  fermi_dirac_deriv(ϵ, T) = df_dT(ϵ, T, μ)
  df_dTs = [fermi_dirac_deriv(ϵ, T) for ϵ in ϵs, T in Ts]

  fig = Figure(size=(800, 600))
  axis = Axis3(
    fig[1, 1],
    xlabel="Energy (J)",
    ylabel="Temperature (K)",
    zlabel="df/dt(ϵ, T)"
  )
  surface!(ϵs, Ts, df_dTs)
  return fig
end

function plot_df_dT_textbook()
  μ = 5e4 * k_B
  return plot_df_dT(μ)
end

function D(N, ϵ, L)
  return (L^3 / (2π^2)) * (2 * m_e / ħ^2)^(3 / 2) * ϵ^(1 / 2)
end

function plot_density_of_states(N, L)
  ϵ_min = 0
  ϵ_max = ϵ_F(N, L) * 1.5
  ϵs = range(ϵ_min, ϵ_max, length=400)
  Ds = [D(N, ϵ, L) / N for ϵ in ϵs]

  fig = Figure(size=(800, 600))
  axis = Axis(
    fig[1, 1],
    xlabel="Energy ϵ (J)",
    ylabel="Normalized density of states: D(ϵ) / N",
    limits=(ϵ_min, ϵ_max, D(N, ϵ_min, L) / N, D(N, ϵ_max, L) / N),
  )
  lines!(axis, ϵs, Ds)
  vlines!(axis, [ϵ_F(N, L)], color=:red, linewidth=1.5, linestyle=:dash, label="ϵ_F")
  return fig
end

# Electron gas specific heat as a function of temperature assuming μ = ϵ_F
function C(T, N, L)
  ϵ_f = ϵ_F(N, L)
  function integrand(ϵ)
    res = (ϵ - ϵ_f) * df_dT(ϵ, T, ϵ_f) * D(N, ϵ, L)
    if isnan(res)
      return 0
    end
    return res
  end
  return quadgk(integrand, 0, ϵ_f)[1]
end

function plot_heat_capacity(N, L)
  T_max = (ϵ_F(N, L) / k_B) * 0.01
  T_min = 0.000001 * T_max
  Ts = range(T_min, T_max, length=400)
  Cs = [C(T, N, L) for T in Ts]

  fig = Figure(size=(800, 600))
  axis = Axis(
    fig[1, 1],
    xlabel="Temperature T (K)",
    ylabel="Heat Capacity C(T) (J/K)",
    limits=(T_min, T_max, 0, maximum(Cs) * 1.1),
  )
  lines!(axis, Ts, Cs)
  autolimits!(axis)
  return fig
end

function plot_verif(N, L)
  C_verif = (T, N, L) -> (1 / 3) * π^2 * D(N, ϵ_F(N, L), L) * k_B^2 * T
  T_max = (ϵ_F(N, L) / k_B) * 0.01
  T_min = T_max * 0.000001
  Ts = range(T_min, T_max, length=400)
  Cs = [C_verif(T, N, L) for T in Ts]

  fig = Figure(size=(800, 600))
  axis = Axis(
    fig[1, 1],
    xlabel="Temperature T (K)",
    ylabel="Heat Capacity C(T) (J/K)",
    limits=(T_min, T_max, 0, maximum(Cs) * 1.1),
  )
  lines!(axis, Ts, Cs)
  autolimits!(axis)
  return fig
end

export k_B, ħ, m_e, ϵ_from_book, f, ϵ_F,
  plot_fermi_dirac, plot_fermi_dirac_textbook, plot_df_dT, plot_df_dT_textbook,
  plot_density_of_states, D, plot_heat_capacity, C, df_dT, plot_verif

end # module ElectronSpecificHeat
