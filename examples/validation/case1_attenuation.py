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


def _shock_metrics_from_probe(
    t: np.ndarray,
    p: np.ndarray,
    baseline_pressure: float,
    gradient_fraction: float = 0.35,
    search_fraction: float = 0.45,
) -> Tuple[float, float, float]:
    """
    Extract incident-shock arrival and strength from a probe trace.

    Technique:
    - compute smoothed dp/dt,
    - identify indices where dp/dt exceeds a fraction of its positive maximum,
    - choose the FIRST such index as incident shock arrival (avoids selecting reflected shock),
    - define shock pressure as local maximum in a short window after arrival.
    """
    if len(t) < 8:
        raise RuntimeError("Probe trace is too short to determine shock metrics.")

    dpdt = np.gradient(p, t)
    kernel = np.ones(7) / 7.0
    dpdt_smooth = np.convolve(dpdt, kernel, mode="same")

    n_search = max(8, int(len(t) * search_fraction))
    dpdt_search = dpdt_smooth[:n_search]
    positive_max = float(np.max(dpdt_search))
    if positive_max <= 0.0:
        raise RuntimeError("No positive pressure front detected at probe.")

    grad_threshold = gradient_fraction * positive_max
    candidate = np.where(dpdt_search >= grad_threshold)[0]
    if len(candidate) == 0:
        raise RuntimeError("No shock crossing found at probe; lower gradient_fraction.")

    i_shock = int(candidate[0])
    i1 = i_shock
    i2 = min(len(p), i_shock + 6)
    shock_pressure = float(np.mean(p[i1:i2]))
    arrival_time = float(t[i_shock])
    attenuation_metric = float(np.log(shock_pressure / baseline_pressure))
    return arrival_time, shock_pressure, attenuation_metric


def main(
    mech_filename: str = PROJECT_DIR / "data/mechanisms/Nitrogen.xml",
    show_results: bool = True,
    results_location: Optional[str] = None,
    t_final: float = 60e-3,
    n_x: int = 1000,
    cfl: float = 0.9,
) -> None:
    ct.add_directory(PROJECT_DIR)
    # provided condtions for Case 1
    Ms = 2.4
    T1 = 292.05
    p1 = 2026.499994
    p2 = 13340.21567
    tFinal = t_final

    # plotting parameters
    fontsize = 11

    # provided geometry
    DDriven = 4.5 * 0.0254
    DDriver = DDriven
    LDriver = 142.0 * 0.0254
    LDriven = 9.73

    # Set up gasses and determine the initial pressures
    u1 = 0.0
    u4 = 0.0  # initially 0 velocity
    gas1 = ct.Solution(mech_filename)
    gas4 = ct.Solution(mech_filename)
    T4 = T1  # assumed
    gas1.TP = T1, p1
    gas4.TP = T4, p1  # use p1 as a place holder
    g1 = gas1.cp / gas1.cv
    g4 = gas4.cp / gas4.cv
    a4oa1 = np.sqrt(g4 / g1 * T4 / T1 * gas1.mean_molecular_weight / gas4.mean_molecular_weight)
    p4 = p2 * (1.0 - (g4 - 1.0) / (g1 + 1.0) / a4oa1 * (Ms - 1.0 / Ms)) ** (-2.0 * g4 / (g4 - 1.0))
    p4 *= 1.05  # account for diaphragm
    gas4.TP = T4, p4

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
    state1 = (gas1, u1)
    state4 = (gas4, u4)
    ssbl = stanShock(
        gas1,
        initializeRiemannProblem=(state4, state1, geometry),
        boundaryConditions=boundaryConditions,
        cfl=cfl,
        outputEvery=100,
        includeBoundaryLayerTerms=True,
        DOuter=D,
        Tw=T1,  # assume wall temperature is in thermal eq. with gas
        dlnAdx=dlnAdx,
    )

    # Three probes in the driven section
    probe_locations: Sequence[float] = (2.0, 5.0, 8.0)
    for i, x_probe in enumerate(probe_locations):
        ssbl.addProbe(x_probe, probeName=f"probe_{i+1}")

    # X-t diagram for pressure
    ssbl.addXTDiagram("pressure", skipSteps=10)

    # Solve
    ssbl.advanceSimulation(tFinal)

    # --- Post-processing: attenuation from multi-probe data ---
    arrivals = []
    shock_pressures = []
    attenuation_values = []

    for probe in ssbl.probes:
        t_probe = np.array(probe.t)
        p_probe = np.array(probe.p)
        arrival_t, p_shock, attn = _shock_metrics_from_probe(t_probe, p_probe, p1)
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

    print("\n==== Shock attenuation metrics (BL model ON) ====")
    print("Probe positions [m]:", x_probe)
    print("Shock arrival times [ms]:", arrivals * 1e3)
    print("Shock pressures [bar]:", shock_pressures / 1e5)
    print("ln(p_shock/p1):", attenuation_values)
    print("Attenuation rate dln(p_shock/p1)/dx [1/m]: %.6f" % attenuation_rate)
    print("Average shock speed from x(t_arrival) [m/s]: %.3f" % x_t_slope)

    # --- Plots ---
    plt.close("all")
    mpl.rcParams["font.size"] = fontsize

    # 3 probe graphs in one figure
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 7.5), sharex=True)
    for i, probe in enumerate(ssbl.probes):
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
        "Case 1 (Boundary Layer ON)\nAttenuation rate dln(p_shock/p1)/dx = %.4f 1/m" % attenuation_rate,
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
            os.path.join(results_location, "case1_attenuation.npz"),
            probe_locations=x_probe,
            probe_arrival_times=arrivals,
            probe_shock_pressures=shock_pressures,
            probe_attenuation=attenuation_values,
            attenuation_rate=attenuation_rate,
            attenuation_intercept=attenuation_intercept,
            shock_speed_average=x_t_slope,
            shock_speed_intercept=x_t_intercept,
            xt_time=np.array(ssbl.XTDiagrams["pressure"].t),
            xt_x=np.array(ssbl.XTDiagrams["pressure"].x),
            xt_pressure=np.array(ssbl.XTDiagrams["pressure"].variable),
        )
        fig.savefig(os.path.join(results_location, "case1_attenuation_probes.png"), dpi=200)
        plt.figure(2)
        plt.savefig(os.path.join(results_location, "case1_attenuation_xt.png"), dpi=200)


if __name__ == "__main__":
    main()
