# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import inspect
import numpy as np

import os
import json
import time
import glob
import shutil
from pathlib import Path
from datetime import datetime

try:
    import pygad
except Exception as e:
    pygad = None
    _PYGAD_IMPORT_ERROR = e

try:
    import plotly.graph_objects as go
except Exception as e:
    go = None
    _PLOTLY_IMPORT_ERROR = e


Number = Union[int, float]
Fitness = Union[Number, Sequence[Number]]


def _ensure_pygad():
    if pygad is None:
        raise ImportError(
            "PyGAD is not installed or failed to import. "
            "Install with: pip install pygad\n"
            f"Original error: {_PYGAD_IMPORT_ERROR}"
        )


def _ensure_plotly():
    if go is None:
        raise ImportError(
            "Plotly is not installed or failed to import. "
            "Install with: pip install plotly\n"
            f"Original error: {_PLOTLY_IMPORT_ERROR}"
        )


def _as_2d_array(x: Any) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Dominance for maximization objectives:
    a dominates b if a >= b in all objectives AND a > b in at least one.
    """
    return np.all(a >= b) and np.any(a > b)


def _pareto_front_indices_max(fitness: np.ndarray) -> np.ndarray:
    """
    Compute indices of the non-dominated set (Pareto front) for maximization.
    Complexity: O(n^2 * m). For typical GA populations this is fine.
    """
    fitness = _as_2d_array(fitness)
    n = fitness.shape[0]
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[i]:
                continue
            if _dominates(fitness[j], fitness[i]):
                is_dominated[i] = True
                break
    return np.where(~is_dominated)[0]

def _pygad_accepts(param_name: str) -> bool:
    try:
        return param_name in inspect.signature(pygad.GA.__init__).parameters
    except Exception:
        return False

def _json_default(obj):
    # Types/classes (e.g. float, int)
    if isinstance(obj, type):
        return obj.__name__
    # Numpy scalars
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    # Callables (functions, lambdas)
    if callable(obj):
        return getattr(obj, "__name__", repr(obj))
    # Fallback: string representation
    return repr(obj)

@dataclass
class GARunSummary:
    generations_completed: int
    best_solution: Optional[np.ndarray]
    best_fitness: Optional[Fitness]
    best_index: Optional[int]


class GA:
    """
    A TopologicPy-style wrapper around PyGAD with ergonomic hyperparameters,
    result accessors, and Plotly Pareto visualizations.

    Key ideas:
    - You provide a fitness function (maximization). PyGAD calls it as:
        fitness(ga_instance, solution, solution_idx) -> float OR sequence of floats
      (single-objective or multi-objective). :contentReference[oaicite:1]{index=1}
    - For multi-objective runs, set parent_selection_type="nsga2" (or another MO selector).
    - This wrapper caches fitness, solutions, and convenient Pareto-front access.

    Minimal usage:
    --------------
    def fitness(ga, sol, sol_idx):
        # maximize something
        return -np.sum((sol - 0.3)**2)

    ga = GA(
        num_genes=5,
        gene_space=[{"low": -1, "high": 1}] * 5,
        fitness_function=fitness,
    )
    ga.Run()
    print(ga.BestFitness, ga.BestSolution)

    Multi-objective usage:
    ----------------------
    def fitness(ga, sol, sol_idx):
        f1 = -np.sum((sol - 0.2)**2)
        f2 = -np.sum((sol + 0.2)**2)
        return [f1, f2]

    ga = GA(
        num_genes=5,
        gene_space=[{"low": -1, "high": 1}] * 5,
        fitness_function=fitness,
        parent_selection_type="nsga2",
        num_parents_mating=20,
    )
    ga.Run()
    fig = ga.PlotParetoFront()
    fig.show()
    """

    def __init__(
        self,
        num_genes: Optional[int] = None,
        gene_space: Optional[Any] = None,
        fitness_function: Optional[Callable[[Any, np.ndarray, int], Fitness]] = None,
        *,
        # Common hyperparameters (sane defaults)
        sol_per_pop: int = 50,
        num_generations: int = 100,
        num_parents_mating: Optional[int] = None,
        parent_selection_type: str = "sss",
        keep_parents: int = 1,
        crossover_type: Union[str, None] = "single_point",
        mutation_type: Union[str, None] = "random",
        mutation_num_genes: Optional[int] = None,
        mutation_percent_genes: Union[int, str] = "default",
        gene_type: Union[type, Sequence[type], None] = float,
        init_range_low: Number = -1.0,
        init_range_high: Number = 1.0,
        allow_duplicate_genes: bool = True,
        random_seed: Optional[int] = None,
        stop_criteria: Optional[Union[str, List[str]]] = None,
        on_generation: Optional[Callable[[Any], Any]] = None,
        # Advanced passthrough
        pygad_kwargs: Optional[Dict[str, Any]] = None,
        silent: bool = False,
    ):
        _ensure_pygad()

        self._silent = silent

        self._fitness_user = fitness_function
        self._gene_space = gene_space
        self._num_genes = num_genes

        self._pygad_kwargs = dict(pygad_kwargs or {})

        # Defaults
        if num_parents_mating is None:
            num_parents_mating = max(2, sol_per_pop // 3)

        self._params: Dict[str, Any] = dict(
            num_genes=num_genes,
            gene_space=gene_space,
            sol_per_pop=sol_per_pop,
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            parent_selection_type=parent_selection_type,
            keep_parents=keep_parents,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_num_genes=mutation_num_genes,
            mutation_percent_genes=mutation_percent_genes,
            gene_type=gene_type,
            init_range_low=init_range_low,
            init_range_high=init_range_high,
            allow_duplicate_genes=allow_duplicate_genes,
            random_seed=random_seed,
            stop_criteria=stop_criteria,
            on_generation=on_generation,
        )

        # Run state
        self._ga: Optional[Any] = None
        self._ran: bool = False

        # Cached results
        self._best_solution: Optional[np.ndarray] = None
        self._best_fitness: Optional[Fitness] = None
        self._best_index: Optional[int] = None
        self._last_population: Optional[np.ndarray] = None
        self._last_population_fitness: Optional[np.ndarray] = None

    def EnableCheckpointing(
        self,
        directory: str,
        interval: int = 5,
        keep_last: int = 5,
        prefix: str = "ga_checkpoint",
        save_final: bool = True,
        atomic: bool = True,
        silent: bool = False,
    ) -> "GA":
        """
        Enable periodic checkpointing.

        Parameters
        ----------
        directory : str
            Directory where checkpoints will be written.
        interval : int, optional
            Save every N generations. Default is 5.
        keep_last : int, optional
            Keep only the most recent K checkpoints. Default is 5.
        prefix : str, optional
            File prefix. Default is "ga_checkpoint".
        save_final : bool, optional
            If True, save a final checkpoint at the end of Run(). Default is True.
        atomic : bool, optional
            If True, write to a temp file then move into place. Default is True.
        silent : bool, optional
            If True, suppress checkpoint log messages. Default is False.
        """
        self._ckpt = {
            "enabled": True,
            "dir": str(directory),
            "interval": max(1, int(interval)),
            "keep_last": max(1, int(keep_last)),
            "prefix": str(prefix),
            "save_final": bool(save_final),
            "atomic": bool(atomic),
            "silent": bool(silent),
        }
        Path(self._ckpt["dir"]).mkdir(parents=True, exist_ok=True)
        self._ckpt["checkpoint_at_gen0"] = False  # or expose as argument
        self._ckpt_files = []  # rolling list of checkpoint .pkl paths (most recent last)
        return self

    def _checkpoint_paths(self) -> list[str]:
        if not getattr(self, "_ckpt", {}).get("enabled", False):
            return []
        pattern = os.path.join(self._ckpt["dir"], f"{self._ckpt['prefix']}*_gen_*.pkl")
        files = glob.glob(pattern)

        def _gen_key(fp: str):
            base = os.path.basename(fp)
            try:
                # handles both: prefix_gen_000123.pkl and prefix_tag_gen_000123.pkl
                g = int(base.split("_gen_")[1].split(".pkl")[0])
                return (0, g)
            except Exception:
                return (1, os.path.getmtime(fp))

        return sorted(files, key=_gen_key)

    def LatestCheckpoint(self) -> str | None:
        files = self._checkpoint_paths()
        return files[-1] if files else None

    def ResumeFromLatestCheckpoint(self) -> bool:
        """
        Load the latest checkpoint (if available) into this GA wrapper.

        Returns
        -------
        bool
            True if a checkpoint was found and loaded, False otherwise.
        """
        import pygad

        latest = self.LatestCheckpoint()
        if latest is None:
            return False

        self._ga = pygad.load(latest)
        self._ran = True

        # Refresh caches if possible
        try:
            sol, fit, idx = self._ga.best_solution()
            self._best_solution = None if sol is None else np.asarray(sol)
            self._best_fitness = fit
            self._best_index = None if idx is None else int(idx)
        except Exception:
            pass

        try:
            self._last_population = np.asarray(getattr(self._ga, "population", None)) if hasattr(self._ga, "population") else None
            lgf = getattr(self._ga, "last_generation_fitness", None)
            self._last_population_fitness = np.asarray(lgf) if lgf is not None else None
        except Exception:
            pass

        return True

    def _save_checkpoint(self, ga_instance, generation: int, tag: str = "") -> None:
        if not getattr(self, "_ckpt", {}).get("enabled", False):
            return
        if ga_instance is None:
            return

        import os
        import json
        from datetime import datetime

        def _json_default(obj):
            if isinstance(obj, type):
                return obj.__name__
            try:
                import numpy as np
                if isinstance(obj, np.generic):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
            except Exception:
                pass
            if callable(obj):
                return getattr(obj, "__name__", repr(obj))
            return repr(obj)

        d = self._ckpt["dir"]
        prefix = self._ckpt["prefix"]
        atomic = self._ckpt["atomic"]
        silent = self._ckpt["silent"]

        os.makedirs(d, exist_ok=True)

        fname = f"{prefix}_gen_{generation:06d}.pkl" if not tag else f"{prefix}_{tag}_gen_{generation:06d}.pkl"
        final_path = os.path.join(d, fname)
        meta_path = final_path[:-4] + ".json"

        meta = {
            "generation": int(generation),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "params": getattr(self, "_params", None),
        }

        try:
            if atomic:
                tmp_base = final_path[:-4] + ".tmp"
                tmp_meta = meta_path + ".tmp"

                ga_instance.save(tmp_base)
                tmp_actual = tmp_base + ".pkl"

                with open(tmp_meta, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, default=_json_default)

                os.replace(tmp_actual, final_path)
                os.replace(tmp_meta, meta_path)
            else:
                ga_instance.save(final_path[:-4])
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, default=_json_default)

        except Exception as e:
            if not silent and not self._silent:
                print(f"[GA] Checkpoint failed at gen {generation}: {e}")
            return

        keep_last = int(self._ckpt.get("keep_last", 5))
        files = self._checkpoint_paths()
        if len(files) > keep_last:
            for fp in files[:-keep_last]:
                try:
                    os.remove(fp)
                except Exception:
                    pass
                try:
                    os.remove(fp[:-4] + ".json")
                except Exception:
                    pass

        if not silent and not self._silent:
            print(f"[GA] Checkpoint saved: {final_path}")


    def _wrap_on_generation(self, user_on_generation):
        """
        Wraps user's on_generation to add checkpointing without breaking their callback.
        """
        def _cb(ga_instance):
            # User callback first (or after) — choose one. I prefer first so it can adjust state.
            if callable(user_on_generation):
                user_on_generation(ga_instance)

            # PyGAD exposes generations_completed
            gen = int(getattr(ga_instance, "generations_completed", 0))

            ck = getattr(self, "_ckpt", None)
            if ck and ck.get("enabled", False):
                interval = ck["interval"]
                if gen > 0 and (gen % interval == 0):
                    self._save_checkpoint(ga_instance, gen)

        return _cb
    # -------------------------
    # Hyperparameter management
    # -------------------------

    def SetParams(self, **kwargs) -> "GA":
        """
        Set/override hyperparameters before calling Run().
        Example: ga.SetParams(num_generations=200, mutation_percent_genes=10)
        """
        for k, v in kwargs.items():
            if k == "pygad_kwargs":
                self._pygad_kwargs.update(v or {})
            else:
                self._params[k] = v
        return self

    def Params(self) -> Dict[str, Any]:
        """Return a shallow copy of current hyperparameters."""
        return dict(self._params, pygad_kwargs=dict(self._pygad_kwargs))

    # -------------------------
    # Building & running PyGAD
    # -------------------------

    def _fitness_proxy(self, ga_instance, solution, solution_idx):
        if self._fitness_user is None:
            raise ValueError("fitness_function is not set.")
        val = self._fitness_user(ga_instance, np.asarray(solution, dtype=float), int(solution_idx))
        return val

    def Build(self) -> "GA":
        """
        Build the internal pygad.GA instance. You normally don't need to call this;
        Run() will call it if needed.
        """
        import warnings
        import inspect

        _ensure_pygad()

        p = dict(self._params)

        # Merge advanced kwargs FIRST (so we can sanitize everything consistently)
        for k, v in self._pygad_kwargs.items():
            if k not in p:
                p[k] = v

        # Defensive: strip deprecated parameter (PyGAD >= 3.3.0)
        p.pop("delay_after_gen", None)

        # Handle mutation_num_genes across PyGAD versions (if you added it to _params)
        mng = p.pop("mutation_num_genes", None)
        if mng is not None:
            if _pygad_accepts("mutation_num_genes"):
                p["mutation_num_genes"] = int(mng)
                # Optional: avoid ambiguity
                p.pop("mutation_percent_genes", None)
            else:
                # Convert to percent that mutates ~mng genes
                ng = int(p.get("num_genes") or 0)
                if ng > 0:
                    percent = int(np.ceil(100.0 * float(mng) / float(ng)))
                    p["mutation_percent_genes"] = max(1, min(100, percent))

        # Basic validation for user friendliness
        if p.get("num_genes") is None:
            raise ValueError("num_genes must be set.")
        if p.get("gene_space") is None:
            raise ValueError("gene_space must be set (list/dict/range as supported by PyGAD).")
        if self._fitness_user is None:
            raise ValueError("fitness_function must be set.")

        # Install checkpointing wrapper around any user on_generation.
        # Make it idempotent so Build() can be safely called multiple times.
        user_cb = p.get("on_generation", None)

        # If already wrapped, unwrap back to the original callback before wrapping again.
        if callable(user_cb) and getattr(user_cb, "_tpga_wrapped", False):
            user_cb = getattr(user_cb, "_tpga_original", None)

        wrapped = self._wrap_on_generation(user_cb)
        # mark wrapper so we can detect it later
        setattr(wrapped, "_tpga_wrapped", True)
        setattr(wrapped, "_tpga_original", user_cb)

        p["on_generation"] = wrapped

        # (Optional but recommended) Only pass parameters PyGAD actually accepts.
        # This makes your wrapper resilient to PyGAD version drift.
        try:
            sig = inspect.signature(pygad.GA.__init__)
            allowed = set(sig.parameters.keys())
            p = {k: v for k, v in p.items() if k in allowed}
        except Exception:
            # If signature introspection fails, fall back to passing p as-is.
            pass

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*'delay_after_gen' parameter is deprecated.*",
                category=UserWarning,
            )
            self._ga = pygad.GA(
                fitness_func=self._fitness_proxy,
                **p,
            )

        self._ran = False
        return self


    def Run(self, target_generations: int = None) -> GARunSummary:
        """
        Execute the genetic algorithm and cache commonly-used results.

        Parameters
        ----------
        target_generations : int, optional
            If provided, ensures the run continues until this TOTAL number of generations
            (useful when resuming from checkpoints). If current generations_completed is already
            >= target_generations, no evolution is run.
        """
        _ensure_pygad()

        if self._ga is None:
            self.Build()

        assert self._ga is not None

        if target_generations is not None:
            done = int(getattr(self._ga, "generations_completed", 0))
            remaining = int(target_generations) - done
            if remaining <= 0:
                # nothing to do, just refresh caches + return summary
                self._ran = True
            else:
                # update GA to run only remaining gens
                # PyGAD stores config on the instance; num_generations is a public attribute in many versions.
                try:
                    self._ga.num_generations = remaining
                except Exception:
                    # If not supported, fall back to running as configured.
                    pass
                self._ga.run()
                self._ran = True
        else:
            self._ga.run()
            self._ran = True

        # Best solution
        sol, fit, idx = self._ga.best_solution()
        self._best_solution = np.asarray(sol) if sol is not None else None
        self._best_fitness = fit
        self._best_index = int(idx) if idx is not None else None

        # Cache last population and fitness
        self._last_population = np.asarray(getattr(self._ga, "population", None)) if hasattr(self._ga, "population") else None
        lgf = getattr(self._ga, "last_generation_fitness", None)
        self._last_population_fitness = np.asarray(lgf) if lgf is not None else None

        # Save final checkpoint if enabled
        ck = getattr(self, "_ckpt", None)
        if ck and ck.get("enabled", False) and ck.get("save_final", True):
            gen = int(getattr(self._ga, "generations_completed", 0))
            self._save_checkpoint(self._ga, gen, tag="final")

        return GARunSummary(
            generations_completed=int(getattr(self._ga, "generations_completed", 0)),
            best_solution=self._best_solution,
            best_fitness=self._best_fitness,
            best_index=self._best_index,
        )

    # -------------------------
    # Easy result accessors
    # -------------------------

    @property
    def GA(self) -> Any:
        """Direct access to the underlying pygad.GA instance (after Build())."""
        return self._ga

    @property
    def Ran(self) -> bool:
        return self._ran

    @property
    def BestSolution(self) -> Optional[np.ndarray]:
        return self._best_solution

    @property
    def BestFitness(self) -> Optional[Fitness]:
        return self._best_fitness

    @property
    def BestIndex(self) -> Optional[int]:
        return self._best_index

    @property
    def Population(self) -> Optional[np.ndarray]:
        """Population of the last generation (if available)."""
        return self._last_population

    @property
    def PopulationFitness(self) -> Optional[np.ndarray]:
        """Fitness array for the last generation (if available)."""
        return self._last_population_fitness

    def Results(self) -> Dict[str, Any]:
        """
        Returns a compact results dictionary designed to be easy to print/log/store.
        """
        gens = int(getattr(self._ga, "generations_completed", 0)) if self._ga is not None else 0
        return {
            "ran": self._ran,
            "generations_completed": gens,
            "best_solution": None if self._best_solution is None else self._best_solution.tolist(),
            "best_fitness": self._best_fitness if self._best_fitness is None else (list(self._best_fitness) if isinstance(self._best_fitness, (list, tuple, np.ndarray)) else self._best_fitness),
            "best_index": self._best_index,
            "population_shape": None if self._last_population is None else list(self._last_population.shape),
            "fitness_shape": None if self._last_population_fitness is None else list(np.asarray(self._last_population_fitness).shape),
            "params": self.Params(),
        }

    # -------------------------
    # Pareto helpers + Plotly
    # -------------------------

    def ParetoFrontIndices(self, fitness: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return indices of non-dominated solutions in the provided fitness array
        (or the cached last-generation fitness).
        Assumes maximization objectives (PyGAD convention). :contentReference[oaicite:3]{index=3}
        """
        if fitness is None:
            if self._last_population_fitness is None:
                raise ValueError("No fitness available. Run() first or pass a fitness array.")
            fitness = self._last_population_fitness
        fitness = _as_2d_array(fitness)
        if fitness.shape[1] < 2:
            raise ValueError("Pareto front requires >= 2 objectives (multi-objective fitness).")
        return _pareto_front_indices_max(fitness)

    def ParetoFront(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (pareto_solutions, pareto_fitness) from the last generation.
        """
        if self._last_population is None or self._last_population_fitness is None:
            raise ValueError("Population/fitness not cached. Run() first.")
        fit = _as_2d_array(self._last_population_fitness)
        idx = self.ParetoFrontIndices(fit)
        return self._last_population[idx], fit[idx]

    def PlotParetoFront(
        self,
        *,
        title: str = "Pareto Front",
        objective_names: list[str] = None,
        show_all_points: bool = True,
        show_pareto_points: bool = True,
        connect_pareto: bool = False,
        pareto_color: str = "#000000",
        population_color: str = "#B0B0B0",
        pareto_size: int = 8,
        population_size: int = 5,
        background_color: str = "white",
        font_family: str = "Arial",
        font_size: int = 14,
        width: int = 800,
        height: int = 600,
        show_grid: bool = True,
        grid_color: str = "#E6E6E6",
        grid_width: float = 1.0,
        show_legend: bool = True,
        max_points: int = None,
    ):
        _ensure_plotly()

        if self._last_population_fitness is None:
            raise ValueError("No fitness cached. Run() first.")

        fit = _as_2d_array(self._last_population_fitness)
        n, m = fit.shape
        if m < 2:
            raise ValueError("PlotParetoFront requires >= 2 objectives.")

        if objective_names is None:
            objective_names = [f"f{i+1}" for i in range(m)]
        if len(objective_names) != m:
            raise ValueError("objective_names length must match number of objectives.")

        indices = np.arange(n)
        if max_points is not None and n > max_points:
            indices = np.random.choice(indices, size=max_points, replace=False)
            fit_viz = fit[indices]
        else:
            fit_viz = fit

        pareto_indices = self.ParetoFrontIndices(fit)
        pareto_mask = np.isin(indices, pareto_indices)

        # ---------- 2D ----------
        if m == 2:
            fig = go.Figure()

            if show_all_points:
                fig.add_trace(
                    go.Scatter(
                        x=fit_viz[:, 0],
                        y=fit_viz[:, 1],
                        mode="markers",
                        marker=dict(size=population_size, color=population_color, line=dict(width=0)),
                        name="Population",
                    )
                )

            if show_pareto_points:
                pareto_x = fit_viz[pareto_mask, 0]
                pareto_y = fit_viz[pareto_mask, 1]

                fig.add_trace(
                    go.Scatter(
                        x=pareto_x,
                        y=pareto_y,
                        mode="markers",
                        marker=dict(size=pareto_size, color=pareto_color, line=dict(width=1, color=pareto_color)),
                        name="Pareto Front",
                    )
                )

                if connect_pareto and len(pareto_x) > 1:
                    order = np.argsort(pareto_x)
                    fig.add_trace(
                        go.Scatter(
                            x=pareto_x[order],
                            y=pareto_y[order],
                            mode="lines",
                            line=dict(color=pareto_color, width=2),
                            showlegend=False,
                        )
                    )

            fig.update_layout(
                title=title,
                width=width,
                height=height,
                plot_bgcolor=background_color,
                paper_bgcolor=background_color,
                font=dict(family=font_family, size=font_size),
                showlegend=show_legend,
                margin=dict(l=60, r=40, t=60, b=60),
            )

            fig.update_xaxes(
                title=objective_names[0],
                showgrid=show_grid,
                gridcolor=grid_color,
                gridwidth=grid_width,
                zeroline=False,
                showline=True,
                linecolor="black",
                mirror=True,
            )

            fig.update_yaxes(
                title=objective_names[1],
                showgrid=show_grid,
                gridcolor=grid_color,
                gridwidth=grid_width,
                zeroline=False,
                showline=True,
                linecolor="black",
                mirror=True,
            )

            return fig

        # ---------- 3D ----------
        if m == 3:
            fig = go.Figure()

            if show_all_points:
                fig.add_trace(
                    go.Scatter3d(
                        x=fit_viz[:, 0],
                        y=fit_viz[:, 1],
                        z=fit_viz[:, 2],
                        mode="markers",
                        marker=dict(size=population_size, color=population_color),
                        name="Population",
                    )
                )

            if show_pareto_points:
                fig.add_trace(
                    go.Scatter3d(
                        x=fit_viz[pareto_mask, 0],
                        y=fit_viz[pareto_mask, 1],
                        z=fit_viz[pareto_mask, 2],
                        mode="markers",
                        marker=dict(size=pareto_size, color=pareto_color),
                        name="Pareto Front",
                    )
                )

            fig.update_layout(
                title=title,
                width=width,
                height=height,
                plot_bgcolor=background_color,
                paper_bgcolor=background_color,
                font=dict(family=font_family, size=font_size),
                showlegend=show_legend,
                scene=dict(
                    xaxis=dict(
                        title=objective_names[0],
                        showgrid=show_grid,
                        gridcolor=grid_color,
                        gridwidth=grid_width,
                        backgroundcolor=background_color,
                        zeroline=False,
                    ),
                    yaxis=dict(
                        title=objective_names[1],
                        showgrid=show_grid,
                        gridcolor=grid_color,
                        gridwidth=grid_width,
                        backgroundcolor=background_color,
                        zeroline=False,
                    ),
                    zaxis=dict(
                        title=objective_names[2],
                        showgrid=show_grid,
                        gridcolor=grid_color,
                        gridwidth=grid_width,
                        backgroundcolor=background_color,
                        zeroline=False,
                    ),
                ),
            )

            return fig

        # ---------- >3 ----------
        dims = [dict(label=objective_names[i], values=fit_viz[:, i]) for i in range(m)]
        color = np.where(pareto_mask, 1, 0)

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=color,
                    colorscale=[[0, population_color], [1, pareto_color]],
                    showscale=False,
                ),
                dimensions=dims,
            )
        )

        fig.update_layout(
            title=title,
            width=width,
            height=height,
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            font=dict(family=font_family, size=font_size),
        )

        return fig

    # -------------------------
    # Saving/loading
    # -------------------------

    def Save(self, filepath: str) -> None:
        """Save the PyGAD instance (requires Build() or Run() beforehand)."""
        if self._ga is None:
            raise ValueError("Nothing to save. Build() or Run() first.")
        self._ga.save(filepath)

    def Load(self, filepath: str) -> "GA":
        """Load a saved PyGAD instance and attach it to this wrapper."""
        _ensure_pygad()
        self._ga = pygad.load(filepath)
        self._ran = True  # loaded instances usually contain run state

        # Try to refresh caches
        try:
            sol, fit, idx = self._ga.best_solution()
            self._best_solution = np.asarray(sol) if sol is not None else None
            self._best_fitness = fit
            self._best_index = int(idx) if idx is not None else None
        except Exception:
            pass

        try:
            self._last_population = np.asarray(getattr(self._ga, "population", None)) if hasattr(self._ga, "population") else None
            lgf = getattr(self._ga, "last_generation_fitness", None)
            self._last_population_fitness = np.asarray(lgf) if lgf is not None else None
        except Exception:
            pass

        return self
