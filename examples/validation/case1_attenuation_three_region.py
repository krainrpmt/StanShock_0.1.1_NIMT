#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    Copyright 2017 Kevin Grogan

    This file is part of StanShock.

    StanShock is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License.

    StanShock is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with StanShock.  If not, see <https://www.gnu.org/licenses/>.
'''
import os.path
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cantera as ct
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from StanShock.stanShock import stanShock

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Note: small no-op comment to force a new commit/PR cycle.

def _shock_metrics_from_probe(
    t: np.ndarray,
    p: np.ndarray,
    baseline_pressure: float,
    rise_fraction: float = 0.03,
) -> Tuple[float, float, float]:
    """
    Extract incident-shock arrival and strength from a probe trace.

    Shock-capturing logic used here:
    1) Estimate probe baseline pressure from the first ~5% of samples.
    2) Compute an adaptive rise threshold as:
         p_threshold = p_baseline + rise_fraction * (p_max - p_baseline)
       This intentionally allows detection of a *small first step* (incident shock)
       even when a larger reflected-shock jump appears later.
    3) Pick the first time index where pressure crosses p_threshold.
    4) Refine arrival to the strongest local dp/dt near that crossing.
    5) Compute shock pressure as the short-window mean right after arrival.
    """
    if len(t) < 10:
        raise RuntimeError("Probe trace is too short to determine shock metrics.")

    n0 = max(5, int(0.05 * len(p)))
    p_baseline = float(np.mean(p[:n0]))
    p_dynamic = float(np.max(p) - p_baseline)
    if p_dynamic <= 0.0:
        arrival_time = float(t[0])
        shock_pressure = p_baseline
        attenuation_metric = float(np.log(max(shock_pressure, 1e-12) / baseline_pressure))
        return arrival_time, shock_pressure, attenuation_metric

    p_threshold = p_baseline + rise_fraction * p_dynamic
    crossing = np.where(p >= p_threshold)[0]
    if len(crossing) == 0:
        i_cross = int(np.argmax(np.gradient(p, t)))
    else:
        i_cross = int(crossing[0])


    i_lo = max(1, i_cross - 5)
    i_hi = min(len(p) - 1, i_cross + 6)
    dpdt_local = np.gradient(p[i_lo:i_hi], t[i_lo:i_hi])
    i_shock = i_lo + int(np.argmax(dpdt_local))

    i1 = i_shock
    i2 = min(len(p), i_shock + 6)
    shock_pressure = float(np.mean(p[i1:i2]))
    arrival_time = float(t[i_shock])
    attenuation_metric = float(np.log(shock_pressure / baseline_pressure))
    return arrival_time, shock_pressure, attenuation_metric

def _gas_sound_speed(gas: ct.Solution) -> float:
    """Return sound speed with compatibility across Cantera versions."""
    if hasattr(gas, "sound_speed"):
        return float(gas.sound_speed)
    if hasattr(gas, "soundspeed"):
        return float(gas.soundspeed)
    gamma = gas.cp_mass / gas.cv_mass
    return float(np.sqrt(gamma * gas.P / gas.density))


def main(
    mech_filename: str = "gri30.yaml",
    show_results: bool = True,
    results_location: Optional[str] = None,
    t_final: float = 60e-3,
    n_x: int = 1000,
    cfl: float = 0.9,
    buffer_test_interface_x: float = 6.0,
    reacting: bool = False,
    # user-specified thermodynamic states
    T1_buffer: float = 292.05,
    P1_buffer: float = 2026.499994,
    T1_test: float = 292.05,
    P1_test: float = 2026.499994,
    T4_driver: float = 292.05,
    P4_driver: float = 25000.0,
    # user-specified compositions
    X_buffer: str = "O2:0.21,AR:0.79",
    X_test: str = "N2:1.0",
    X_driver: str = "H2:0.25,N2:0.75",
) -> None:
    ct.add_directory(PROJECT_DIR)
    # simulation controls
    tFinal = t_final

    # plotting parameters
    fontsize = 11

    # provided geometry
    DDriven = 4.5 * 0.0254
    DDriver = DDriven
    LDriver = 142.0 * 0.0254
    LDriven = 9.73

    # Set up gases and determine the initial pressures
    # Region naming: driver (4), buffer (1b), test/CRV (1t)
    u1 = 0.0
    u4 = 0.0  # initially 0 velocity
    gas_buffer = ct.Solution(mech_filename)
    gas_test = ct.Solution(mech_filename)
    gas_driver = ct.Solution(mech_filename)

    # User-specified states (no P4/Mach inference)
    gas_buffer.TPX = T1_buffer, P1_buffer, X_buffer
    gas_test.TPX = T1_test, P1_test, X_test
    gas_driver.TPX = T4_driver, P4_driver, X_driver

    # set up geometry
    nX = n_x  # mesh resolution
    xLower = -LDriver
    xUpper = LDriven
    xShock = 0.0
    geometry = (nX, xLower, xUpper, xShock)
    DeltaD = DDriven - DDriver
    DeltaX = (xUpper - xLower) / float(nX) * 10  # diffuse area change for numerical stability

    def D(x):
        diameter = DDriven + (DeltaD / DeltaX) * (x - xShock)
        diameter[x < (xShock - DeltaX)] = DDriver
        diameter[x > xShock] = DDriven
        return diameter

    def dDdx(x):
        dDiameterdx = np.ones(len(x)) * (DeltaD / DeltaX)
        dDiameterdx[x < (xShock - DeltaX)] = 0.0
        dDiameterdx[x > xShock] = 0.0
        return dDiameterdx

    A = lambda x: np.pi / 4.0 * D(x) ** 2.0
    dAdx = lambda x: np.pi / 2.0 * D(x) * dDdx(x)
    dlnAdx = lambda x, t: dAdx(x) / A(x)

    # Turn ON boundary-layer model as requested
    print("Solving with boundary layer terms")
    boundaryConditions = ["reflecting", "reflecting"]
    state1 = (gas_buffer, u1)
    state4 = (gas_driver, u4)
    ssbl = stanShock(
        gas_buffer,
        initializeRiemannProblem=(state4, state1, geometry),
        boundaryConditions=boundaryConditions,
        cfl=cfl,
        outputEvery=100,
        includeBoundaryLayerTerms=True,
        DOuter=D,
        Tw=T1_buffer,  # assume wall temperature near driven initial gas temperature
        dlnAdx=dlnAdx,
    )

    # --- Overwrite initial condition to create THREE regions: driver | buffer | test ---
    # x < 0: driver, 0 <= x < interface: buffer, x >= interface: test (CRV)
    if buffer_test_interface_x <= 0.0 or buffer_test_interface_x >= LDriven:
        raise ValueError("buffer_test_interface_x must be inside driven section: 0 < x < LDriven")

    driver_idx = ssbl.x < 0.0
    buffer_idx = np.logical_and(ssbl.x >= 0.0, ssbl.x < buffer_test_interface_x)
    test_idx = ssbl.x >= buffer_test_interface_x

    ssbl.r[driver_idx] = gas_driver.density
    ssbl.u[driver_idx] = u4
    ssbl.p[driver_idx] = gas_driver.P
    ssbl.gamma[driver_idx] = gas_driver.cp / gas_driver.cv
    ssbl.Y[driver_idx, :] = gas_driver.Y

    ssbl.r[buffer_idx] = gas_buffer.density
    ssbl.u[buffer_idx] = u1
    ssbl.p[buffer_idx] = gas_buffer.P
    ssbl.gamma[buffer_idx] = gas_buffer.cp / gas_buffer.cv
    ssbl.Y[buffer_idx, :] = gas_buffer.Y

    ssbl.r[test_idx] = gas_test.density
    ssbl.u[test_idx] = u1
    ssbl.p[test_idx] = gas_test.P
    ssbl.gamma[test_idx] = gas_test.cp / gas_test.cv
    ssbl.Y[test_idx, :] = gas_test.Y

    # Optional chemistry in test section only
    ssbl.reacting = reacting
    ssbl.inReactingRegion = lambda x, t: x >= buffer_test_interface_x

    # Three probes in the driven section (for attenuation)
    probe_locations: Sequence[float] = (2.0, 5.0, 8.0)
    for i, x_probe in enumerate(probe_locations):
        ssbl.addProbe(x_probe, probeName=f"probe_{i+1}")
    ssbl.addProbe(max(ssbl.x), probeName="endwall")

    # X-t diagram for pressure
    ssbl.addXTDiagram("pressure", skipSteps=10)

    # Solve
    ssbl.advanceSimulation(tFinal)

    # --- Post-processing: attenuation from multi-probe data ---
    arrivals = []
    shock_pressures = []
    attenuation_values = []

    for probe in ssbl.probes[:len(probe_locations)]:
        t_probe = np.array(probe.t)
        p_probe = np.array(probe.p)
        arrival_t, p_shock, attn = _shock_metrics_from_probe(t_probe, p_probe, P1_buffer)
        arrivals.append(arrival_t)
        shock_pressures.append(p_shock)
        attenuation_values.append(attn)

    x_probe = np.array(probe_locations)
    arrivals = np.array(arrivals)
    shock_pressures = np.array(shock_pressures)
    attenuation_values = np.array(attenuation_values)

    # attenuation rate based on ln(p_shock/p1) vs x
    attenuation_rate, attenuation_intercept = np.polyfit(x_probe, attenuation_values, 1)

    # average shock speed from x(t_arrival)
    x_t_slope, x_t_intercept = np.polyfit(arrivals, x_probe, 1)

    # shock-speed attenuation from probe-to-probe travel times
    dt_seg = np.diff(arrivals)
    dx_seg = np.diff(x_probe)
    valid_seg = dt_seg > 0.0
    if np.count_nonzero(valid_seg) >= 2:
        x_seg = 0.5 * (x_probe[:-1][valid_seg] + x_probe[1:][valid_seg])
        us_seg = dx_seg[valid_seg] / dt_seg[valid_seg]
        us_attenuation_rate, us_intercept = np.polyfit(x_seg, us_seg, 1)

        a_buffer = _gas_sound_speed(gas_buffer)
        a_test = _gas_sound_speed(gas_test)
        a_probe = np.where(x_probe < buffer_test_interface_x, a_buffer, a_test)
        a_seg = 0.5 * (a_probe[:-1][valid_seg] + a_probe[1:][valid_seg])
        ms_seg = us_seg / a_seg
        ms_attenuation_rate, ms_intercept = np.polyfit(x_seg, ms_seg, 1)
    else:
        us_seg = np.array([])
        ms_seg = np.array([])
        us_attenuation_rate = np.nan
        us_intercept = np.nan
        ms_attenuation_rate = np.nan
        ms_intercept = np.nan

    print("\n==== Shock attenuation metrics (BL model ON) ====")
    print("Probe positions [m]:", x_probe)
    print("Shock arrival times [ms]:", arrivals * 1e3)
    print("Shock pressures [bar]:", shock_pressures / 1e5)
    print("ln(p_shock/p1):", attenuation_values)
    print("Attenuation rate dln(p_shock/P1_buffer)/dx [1/m]: %.6f" % attenuation_rate)
    print("Average shock speed from x(t_arrival) [m/s]: %.3f" % x_t_slope)
    if np.isfinite(us_attenuation_rate):
        print("Shock-speed attenuation dU_s/dx [(m/s)/m]: %.6f" % us_attenuation_rate)
    else:
        print("Shock-speed attenuation dU_s/dx [(m/s)/m]: unavailable (insufficient valid probe segments)")
    if np.isfinite(ms_attenuation_rate):
        print("Mach attenuation dM_s/dx [1/m]: %.6f" % ms_attenuation_rate)
    else:
        print("Mach attenuation dM_s/dx [1/m]: unavailable (insufficient valid probe segments)")

    # Pressure-spike metrics useful for end-wall design
    # Domain maximum is taken over the full X-t pressure history.
    p_xt = np.array(ssbl.XTDiagrams["pressure"].variable)
    p_max_domain = float(np.max(p_xt))
    p_max_endwall = float(np.max(np.array(ssbl.probes[-1].p)))
    print("Maximum pressure over full X-t domain [bar]: %.4f" % (p_max_domain / 1.0e5))
    print("Maximum pressure at end-wall probe [bar]: %.4f" % (p_max_endwall / 1.0e5))

    # --- Plots ---
    plt.close("all")
    mpl.rcParams["font.size"] = fontsize

    # 3 probe graphs in one figure
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 7.5), sharex=True)
    for i, probe in enumerate(ssbl.probes[:len(probe_locations)]):
        t_probe = np.array(probe.t)
        p_probe = np.array(probe.p)
        axes[i].plot(t_probe * 1000.0, p_probe / 1.0e5, "k", linewidth=1.7)
        axes[i].axvline(arrivals[i] * 1000.0, color="r", linestyle="--", linewidth=1.2, label="shock arrival")
        axes[i].set_ylabel("p [bar]")
        axes[i].set_title("Probe %d at x = %.2f m" % (i + 1, probe_locations[i]))
        axes[i].grid(alpha=0.25)
        axes[i].legend(loc="upper right")
    axes[-1].set_xlabel("t [ms]")
    fig.suptitle(
        "Case 1 three-region (Boundary Layer ON)\nAttenuation rate dln(p_shock/P1_buffer)/dx = %.4f 1/m" % attenuation_rate,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # X-t diagram for pressure
    ssbl.plotXTDiagram(ssbl.XTDiagrams["pressure"])
    plt.title("Pressure X-t diagram (Boundary Layer ON)")
    plt.tight_layout()

    if show_results:
        plt.show()

    if results_location is not None:
        np.savez(
            os.path.join(results_location, "case1_attenuation_three_region.npz"),
            probe_locations=x_probe,
            probe_arrival_times=arrivals,
            probe_shock_pressures=shock_pressures,
            probe_attenuation=attenuation_values,
            p1_buffer=P1_buffer,
            p1_test=P1_test,
            p4_driver=P4_driver,
            attenuation_rate=attenuation_rate,
            attenuation_intercept=attenuation_intercept,
            shock_speed_average=x_t_slope,
            shock_speed_intercept=x_t_intercept,
            shock_speed_segment=us_seg,
            shock_speed_attenuation_rate=us_attenuation_rate,
            shock_speed_attenuation_intercept=us_intercept,
            shock_mach_segment=ms_seg,
            shock_mach_attenuation_rate=ms_attenuation_rate,
            shock_mach_attenuation_intercept=ms_intercept,
            p_max_domain=p_max_domain,
            p_max_endwall=p_max_endwall,
            buffer_test_interface_x=buffer_test_interface_x,
            xt_time=np.array(ssbl.XTDiagrams["pressure"].t),
            xt_x=np.array(ssbl.XTDiagrams["pressure"].x),
            xt_pressure=np.array(ssbl.XTDiagrams["pressure"].variable),
        )
        fig.savefig(os.path.join(results_location, "case1_attenuation_three_region_probes.png"), dpi=200)
        plt.figure(2)
        plt.savefig(os.path.join(results_location, "case1_attenuation_three_region_xt.png"), dpi=200)


if __name__ == "__main__":
    main()
