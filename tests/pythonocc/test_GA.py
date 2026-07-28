# Auto-generated PythonOCC backend parity test
# Copied from test_GA.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.GA.

These tests avoid a hard dependency on PyGAD and Plotly by monkeypatching the
small API surface that GA.py uses. They are intended to run deterministically
across the TopologicPy CI matrix without skips or xfails.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import topologicpy.GA as ga_module
from topologicpy.GA import (
    GA,
    GARunSummary,
    _GenerationCallback,
    _as_2d_array,
    _dominates,
    _json_default,
    _pareto_front_indices_max,
    _pygad_base_filename,
    _safe_int,
    _to_json_value,
)


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


class FakeGA:
    """Small deterministic stand-in for pygad.GA."""

    constructed = []

    def __init__(
        self,
        fitness_func,
        num_genes=None,
        gene_space=None,
        sol_per_pop=50,
        num_generations=100,
        num_parents_mating=None,
        parent_selection_type="sss",
        keep_parents=1,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes="default",
        gene_type=float,
        init_range_low=-1.0,
        init_range_high=1.0,
        allow_duplicate_genes=True,
        random_seed=None,
        stop_criteria=None,
        on_generation=None,
    ):
        self.fitness_func = fitness_func
        self.num_genes = int(num_genes) if num_genes is not None else None
        self.gene_space = gene_space
        self.sol_per_pop = int(sol_per_pop)
        self.num_generations = int(num_generations)
        self.num_parents_mating = num_parents_mating
        self.parent_selection_type = parent_selection_type
        self.keep_parents = keep_parents
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.mutation_percent_genes = mutation_percent_genes
        self.gene_type = gene_type
        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
        self.allow_duplicate_genes = allow_duplicate_genes
        self.random_seed = random_seed
        self.stop_criteria = stop_criteria
        self.on_generation = on_generation
        self.generations_completed = 0
        self.callback_returns = []
        self.run_calls = 0

        n_genes = int(self.num_genes or 1)
        base = np.arange(self.sol_per_pop * n_genes, dtype=float).reshape(self.sol_per_pop, n_genes)
        self.population = base / max(1, n_genes)
        self.last_generation_fitness = None
        self.params = {
            "num_genes": self.num_genes,
            "gene_space": self.gene_space,
            "sol_per_pop": self.sol_per_pop,
            "num_generations": self.num_generations,
            "num_parents_mating": self.num_parents_mating,
            "parent_selection_type": self.parent_selection_type,
            "keep_parents": self.keep_parents,
            "crossover_type": self.crossover_type,
            "mutation_type": self.mutation_type,
            "mutation_percent_genes": self.mutation_percent_genes,
            "gene_type": self.gene_type,
            "init_range_low": self.init_range_low,
            "init_range_high": self.init_range_high,
            "allow_duplicate_genes": self.allow_duplicate_genes,
            "random_seed": self.random_seed,
            "stop_criteria": self.stop_criteria,
            "on_generation": self.on_generation,
        }
        FakeGA.constructed.append(self)

    def _evaluate(self):
        values = []
        for i, sol in enumerate(self.population):
            values.append(self.fitness_func(self, sol, i))
        self.last_generation_fitness = np.asarray(values)
        return self.last_generation_fitness

    def run(self):
        self.run_calls += 1
        for _ in range(int(self.num_generations)):
            self.generations_completed += 1
            result = None
            if callable(self.on_generation):
                result = self.on_generation(self)
                self.callback_returns.append(result)
            if result == "stop":
                break
        self._evaluate()

    def best_solution(self):
        fitness = self.last_generation_fitness
        if fitness is None:
            fitness = self._evaluate()
        arr = np.asarray(fitness)
        if arr.ndim <= 1:
            idx = int(np.argmax(arr))
            best_fit = arr[idx].item() if hasattr(arr[idx], "item") else arr[idx]
        else:
            idx = int(np.argmax(np.sum(arr, axis=1)))
            best_fit = arr[idx].tolist()
        return self.population[idx].copy(), best_fit, idx

    def save(self, filepath):
        base = str(filepath)
        if base.lower().endswith(".pkl"):
            base = base[:-4]
        payload = {
            "num_genes": self.num_genes,
            "gene_space": self.gene_space,
            "sol_per_pop": self.sol_per_pop,
            "num_generations": self.num_generations,
            "generations_completed": self.generations_completed,
            "population": self.population.tolist(),
            "last_generation_fitness": None
            if self.last_generation_fitness is None
            else np.asarray(self.last_generation_fitness).tolist(),
        }
        Path(base).parent.mkdir(parents=True, exist_ok=True)
        with open(base + ".pkl", "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, filepath):
        base = str(filepath)
        if base.lower().endswith(".pkl"):
            base = base[:-4]
        with open(base + ".pkl", "r", encoding="utf-8") as f:
            payload = json.load(f)
        fake = cls(
            fitness_func=lambda _ga, sol, _idx: float(np.sum(sol)),
            num_genes=payload.get("num_genes"),
            gene_space=payload.get("gene_space") or [{"low": 0, "high": 1}],
            sol_per_pop=payload.get("sol_per_pop", 2),
            num_generations=payload.get("num_generations", 1),
        )
        fake.generations_completed = int(payload.get("generations_completed", 0))
        fake.population = np.asarray(payload.get("population", fake.population), dtype=float)
        fitness = payload.get("last_generation_fitness", None)
        fake.last_generation_fitness = None if fitness is None else np.asarray(fitness)
        return fake


@pytest.fixture
def fake_pygad(monkeypatch):
    FakeGA.constructed = []
    fake_module = SimpleNamespace(GA=FakeGA, load=FakeGA.load)
    monkeypatch.setattr(ga_module, "pygad", fake_module)
    monkeypatch.setattr(ga_module, "_PYGAD_IMPORT_ERROR", None)
    return fake_module


class FakeTrace:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeFigure:
    def __init__(self, data=None):
        self.traces = []
        self.layout_updates = []
        self.xaxis_updates = []
        self.yaxis_updates = []
        if data is not None:
            if isinstance(data, (list, tuple)):
                self.traces.extend(data)
            else:
                self.traces.append(data)

    def add_trace(self, trace):
        self.traces.append(trace)

    def update_layout(self, **kwargs):
        self.layout_updates.append(kwargs)

    def update_xaxes(self, **kwargs):
        self.xaxis_updates.append(kwargs)

    def update_yaxes(self, **kwargs):
        self.yaxis_updates.append(kwargs)


@pytest.fixture
def fake_plotly(monkeypatch):
    fake_go = SimpleNamespace(
        Figure=FakeFigure,
        Scatter=FakeTrace,
        Scatter3d=FakeTrace,
        Parcoords=FakeTrace,
    )
    monkeypatch.setattr(ga_module, "go", fake_go)
    monkeypatch.setattr(ga_module, "_PLOTLY_IMPORT_ERROR", None)
    return fake_go


def _fitness_single(_ga, solution, _idx):
    return float(np.sum(solution))


def _fitness_multi(_ga, solution, _idx):
    return [float(solution[0]), float(-solution[0])]


def test_array_helpers_and_json_helpers():
    assert _as_2d_array(3).shape == (1, 1)
    assert _as_2d_array([1, 2, 3]).shape == (3, 1)
    assert _as_2d_array([[1, 2], [3, 4]]).shape == (2, 2)

    assert bool(_dominates(np.asarray([2, 2]), np.asarray([1, 2]))) is True
    assert bool(_dominates(np.asarray([1, 2]), np.asarray([2, 1]))) is False

    front = _pareto_front_indices_max(np.asarray([[1, 1], [2, 0], [0, 2], [2, 2]]))
    assert front.tolist() == [3]

    assert _pygad_base_filename("model.pkl").endswith("model")
    assert _pygad_base_filename("model") == "model"
    assert _safe_int("7", 0) == 7
    assert _safe_int("bad", 5) == 5
    assert _json_default(np.asarray([1, 2])) == [1, 2]
    assert _to_json_value({"a": np.int64(3), "b": np.asarray([1.0])}) == {"a": 3, "b": [1.0]}


def test_missing_pygad_raises(monkeypatch):
    monkeypatch.setattr(ga_module, "pygad", None)
    monkeypatch.setattr(ga_module, "_PYGAD_IMPORT_ERROR", RuntimeError("missing"))
    with pytest.raises(ImportError):
        GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_single)


def test_build_validates_required_inputs(fake_pygad):
    with pytest.raises(ValueError):
        GA(num_genes=None, gene_space=[0, 1], fitness_function=_fitness_single).Build()
    with pytest.raises(ValueError):
        GA(num_genes=2, gene_space=None, fitness_function=_fitness_single).Build()
    with pytest.raises(ValueError):
        GA(num_genes=2, gene_space=[0, 1], fitness_function=None).Build()
    with pytest.raises(ValueError):
        GA(num_genes=0, gene_space=[0, 1], fitness_function=_fitness_single).Build()


def test_build_filters_kwargs_converts_mutation_num_genes_and_wraps_callback(fake_pygad):
    seen = []

    def user_callback(ga_instance):
        seen.append(ga_instance.generations_completed)
        return "stop"

    ga = GA(
        num_genes=3,
        gene_space=[{"low": 0, "high": 1}] * 3,
        fitness_function=_fitness_single,
        sol_per_pop=4,
        num_generations=5,
        mutation_num_genes=1,
        on_generation=user_callback,
        pygad_kwargs={"fitness_func": "should_be_removed", "delay_after_gen": 99, "unknown_kw": 123},
    )
    ga.Build()

    assert isinstance(ga.GA.on_generation, _GenerationCallback)
    assert getattr(ga.GA.on_generation, "_tpga_wrapped", False) is True
    assert ga.GA.on_generation(ga.GA) == "stop"
    assert seen == [0]
    assert ga.GA.mutation_percent_genes == 34
    assert not hasattr(ga.GA, "unknown_kw")


def test_run_caches_results_and_preserves_callback_stop(fake_pygad):
    calls = []

    def stop_after_first(ga_instance):
        calls.append(ga_instance.generations_completed)
        return "stop"

    ga = GA(
        num_genes=2,
        gene_space=[{"low": 0, "high": 1}] * 2,
        fitness_function=_fitness_single,
        sol_per_pop=4,
        num_generations=10,
        on_generation=stop_after_first,
    )
    summary = ga.Run()

    assert isinstance(summary, GARunSummary)
    assert summary.generations_completed == 1
    assert ga.Ran is True
    assert calls == [1]
    assert ga.GA.callback_returns == ["stop"]
    assert ga.BestSolution is not None
    assert ga.BestFitness is not None
    assert isinstance(ga.BestIndex, int)
    results = ga.Results()
    assert results["ran"] is True
    assert isinstance(results["best_solution"], list)
    assert results["population_shape"] == [4, 2]


def test_setparams_invalidates_built_instance_and_rebuilds_on_run(fake_pygad):
    ga = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_single, num_generations=1)
    ga.Build()
    first_internal = ga.GA
    ga.SetParams(num_generations=3)

    assert ga.GA is None
    assert ga.Ran is False

    summary = ga.Run()
    assert ga.GA is not first_internal
    assert ga.GA.num_generations == 3
    assert summary.generations_completed == 3


def test_setparams_rejects_invalid_pygad_kwargs(fake_pygad):
    ga = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_single)
    with pytest.raises(ValueError):
        ga.SetParams(pygad_kwargs="not a dict")


def test_run_target_generations(fake_pygad):
    ga = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_single, num_generations=10)

    first = ga.Run(target_generations=2)
    assert first.generations_completed == 2
    assert ga.GA.run_calls == 1

    second = ga.Run(target_generations=1)
    assert second.generations_completed == 2
    assert ga.GA.run_calls == 1

    with pytest.raises(ValueError):
        ga.Run(target_generations=-1)
    with pytest.raises(ValueError):
        ga.Run(target_generations="bad")


def test_save_and_load_accept_pkl_or_base_path(fake_pygad, tmp_path):
    ga = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_single, num_generations=1)
    ga.Run()

    save_path = tmp_path / "saved_ga.pkl"
    ga.Save(str(save_path))
    assert save_path.exists()

    loaded = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_single)
    out = loaded.Load(str(save_path))

    assert out is loaded
    assert loaded.Ran is True
    assert loaded.BestSolution is not None
    assert loaded.Population is not None
    assert loaded.PopulationFitness is not None


def test_save_without_build_or_run_raises(fake_pygad, tmp_path):
    ga = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_single)
    with pytest.raises(ValueError):
        ga.Save(str(tmp_path / "missing"))


def test_checkpointing_latest_resume_and_keep_last(fake_pygad, tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    ga = GA(
        num_genes=2,
        gene_space=[0, 1],
        fitness_function=_fitness_single,
        sol_per_pop=3,
        num_generations=4,
        silent=True,
    )
    ga.EnableCheckpointing(str(ckpt_dir), interval=1, keep_last=2, save_final=True, silent=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        summary = ga.Run()

    files = ga._checkpoint_paths()
    assert summary.generations_completed == 4
    assert 1 <= len(files) <= 2
    assert all(fp.endswith(".pkl") for fp in files)
    assert ga.LatestCheckpoint() == files[-1]

    resumed = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_single)
    resumed.EnableCheckpointing(str(ckpt_dir), interval=1, keep_last=2, silent=True)
    assert resumed.ResumeFromLatestCheckpoint() is True
    assert resumed.Ran is True
    assert resumed.BestSolution is not None


def test_resume_returns_false_when_no_checkpoint(fake_pygad, tmp_path):
    ga = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_single)
    ga.EnableCheckpointing(str(tmp_path / "empty"), silent=True)
    assert ga.LatestCheckpoint() is None
    assert ga.ResumeFromLatestCheckpoint() is False


def test_fitness_proxy_converts_solution_to_float_array(fake_pygad):
    seen = {}

    def fitness(_ga, solution, solution_idx):
        seen["dtype"] = solution.dtype.kind
        seen["idx"] = solution_idx
        return float(np.sum(solution))

    ga = GA(num_genes=2, gene_space=[0, 1], fitness_function=fitness)
    value = ga._fitness_proxy(object(), [1, 2], 7)

    assert value == 3.0
    assert seen == {"dtype": "f", "idx": 7}


def test_pareto_front_indices_and_front(fake_pygad):
    ga = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_multi)
    ga._last_population = np.asarray([[0, 0], [1, 0], [0, 1], [2, 2]], dtype=float)
    ga._last_population_fitness = np.asarray([[1, 1], [2, 0], [0, 2], [2, 2]], dtype=float)

    assert ga.ParetoFrontIndices().tolist() == [3]
    pop, fit = ga.ParetoFront()
    assert pop.tolist() == [[2.0, 2.0]]
    assert fit.tolist() == [[2.0, 2.0]]

    with pytest.raises(ValueError):
        ga.ParetoFrontIndices(np.asarray([1, 2, 3]))

    ga._last_population = np.asarray([[0, 0]], dtype=float)
    ga._last_population_fitness = np.asarray([[1, 1], [2, 2]], dtype=float)
    with pytest.raises(ValueError):
        ga.ParetoFront()


def test_pareto_helpers_require_cached_data(fake_pygad):
    ga = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_multi)
    with pytest.raises(ValueError):
        ga.ParetoFrontIndices()
    with pytest.raises(ValueError):
        ga.ParetoFront()


def test_plot_pareto_front_2d(fake_pygad, fake_plotly):
    ga = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_multi)
    ga._last_population_fitness = np.asarray([[1, 1], [2, 0], [0, 2], [2, 2]], dtype=float)

    fig = ga.PlotParetoFront(connect_pareto=True, max_points=3)

    assert isinstance(fig, FakeFigure)
    assert len(fig.traces) >= 2
    assert fig.layout_updates[-1]["title"] == "Pareto Front"
    assert fig.xaxis_updates[-1]["title"] == "f1"
    assert fig.yaxis_updates[-1]["title"] == "f2"


def test_plot_pareto_front_3d_and_parallel(fake_pygad, fake_plotly):
    ga = GA(num_genes=3, gene_space=[0, 1], fitness_function=_fitness_multi)
    ga._last_population_fitness = np.asarray([[1, 1, 0], [2, 0, 1], [0, 2, 1], [2, 2, 2]], dtype=float)
    fig3 = ga.PlotParetoFront(objective_names=["a", "b", "c"])
    assert isinstance(fig3, FakeFigure)
    assert len(fig3.traces) == 2

    ga._last_population_fitness = np.asarray([[1, 1, 0, 0], [2, 0, 1, 1], [0, 2, 1, 0], [2, 2, 2, 2]], dtype=float)
    fig4 = ga.PlotParetoFront(objective_names=["a", "b", "c", "d"])
    assert isinstance(fig4, FakeFigure)
    assert len(fig4.traces) == 1


def test_plot_pareto_front_validation(fake_pygad, fake_plotly):
    ga = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_multi)

    with pytest.raises(ValueError):
        ga.PlotParetoFront()

    ga._last_population_fitness = np.asarray([1, 2, 3], dtype=float)
    with pytest.raises(ValueError):
        ga.PlotParetoFront()

    ga._last_population_fitness = np.asarray([[1, 1], [2, 2]], dtype=float)
    with pytest.raises(ValueError):
        ga.PlotParetoFront(objective_names=["only_one"])
    with pytest.raises(ValueError):
        ga.PlotParetoFront(max_points=0)
    with pytest.raises(ValueError):
        ga.PlotParetoFront(max_points="bad")


def test_missing_plotly_raises(fake_pygad, monkeypatch):
    ga = GA(num_genes=2, gene_space=[0, 1], fitness_function=_fitness_multi)
    ga._last_population_fitness = np.asarray([[1, 1], [2, 2]], dtype=float)
    monkeypatch.setattr(ga_module, "go", None)
    monkeypatch.setattr(ga_module, "_PLOTLY_IMPORT_ERROR", RuntimeError("missing"))
    with pytest.raises(ImportError):
        ga.PlotParetoFront()
