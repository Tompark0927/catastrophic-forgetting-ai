"""
fza_evolution_arena.py — Synthetic Natural Selection Engine (v20.0)
====================================================================
The Evolution Arena. The survival gauntlet. The engine of digital speciation.

When the Jellyfish Protocol reincarnates FZA, it produces ONE clean clone.
But biology teaches us that survival requires VARIATION + SELECTION.

This module extends the Jellyfish Protocol:
  1. MUTATION: Reads the Core Fact Seed and generates N mutated variants
     by injecting slight perturbations into hyperparameters and Root Axioms.
  2. ARENA: Spawns each variant as an isolated subprocess with a unique
     mutation config, then runs them through a benchmark gauntlet.
  3. SELECTION: The variant with the highest combined score (speed + accuracy
     + memory efficiency) is declared the winner and becomes the new Main FZA.
     All others are killed.

Biological Metaphor:
  Galapagos finches. When a volcanic island is colonized by finches,
  thousands of slight variations compete. The variation with the beak
  perfectly adapted to the available food source outcompetes the others.
  FZA's "beak" is its hyperparameter set (temperature, EWC rate, memory
  pruning aggressiveness). The "food source" is the benchmark gauntlet.

  Unlike standard Darwinian evolution (generational timescales), FZA
  runs this entire cycle in minutes — synthetic punctuated equilibrium.

Usage:
    arena = EvolutionArena(seed_path='./jellyfish_seed/core_seed.json')
    winner = arena.run(population=5)
    print(f"Generation winner: {winner['mutation_id']}")
"""

import os
import json
import time
import copy
import random
import subprocess
import threading
from typing import Optional

from fza_event_bus import bus

# ── Default Hyperparameter Bounds for Mutation ────────────────────────────────
MUTABLE_PARAMS = {
    "temperature":        (0.4,  0.95,  0.05),  # (min, max, std_dev)
    "ewc_lambda":         (0.1,  5.0,   0.5),
    "memory_prune_alpha": (0.05, 0.40,  0.03),
    "reflex_threshold":   (0.3,  0.85,  0.05),
    "lobe_spawn_budget":  (1,    8,     1),
}

BENCHMARK_TIMEOUT_S = 30    # Max seconds any clone can run a benchmark
DEFAULT_POPULATION  = 5     # Number of variants per generation
BENCHMARK_SCRIPT    = os.path.join(os.path.dirname(__file__), "fza_arena_benchmark.py")


class MutationEngine:
    """
    Takes a Core Fact Seed and produces N mutated variants.
    Each variant has a unique mutation_id and slightly different hyperparameters.
    """

    def __init__(self, rng_seed: Optional[int] = None):
        self._rng = random.Random(rng_seed)

    def mutate(self, base_seed: dict, n: int = DEFAULT_POPULATION) -> list[dict]:
        """
        Generate N mutated variants of the base seed.
        Each variant differs in one or more hyperparameters.
        """
        variants = []
        for i in range(n):
            variant = copy.deepcopy(base_seed)
            variant["mutation_id"] = f"clone_{i:02d}"
            variant["mutation_params"] = self._sample_mutation()
            # Slightly shuffle root axioms order to vary domain priority
            if variant.get("root_axioms"):
                axioms = list(variant["root_axioms"])
                self._rng.shuffle(axioms)
                variant["root_axioms"] = axioms
            variant["parent_generation"] = base_seed.get("generation", 0)
            variants.append(variant)
        return variants

    def _sample_mutation(self) -> dict:
        """Sample a random mutation from the hyperparameter bounds."""
        mutation = {}
        for param, (mn, mx, std) in MUTABLE_PARAMS.items():
            base = (mn + mx) / 2
            value = base + self._rng.gauss(0, std)
            value = max(mn, min(mx, value))
            if isinstance(mn, int):
                value = int(round(value))
            mutation[param] = round(value, 4)
        return mutation


class BenchmarkResult:
    """Holds the results of one clone's benchmark run."""
    def __init__(self, mutation_id: str):
        self.mutation_id   = mutation_id
        self.logic_score   = 0.0    # 0.0 – 1.0
        self.speed_score   = 0.0    # 0.0 – 1.0, faster = higher
        self.memory_score  = 0.0    # 0.0 – 1.0, leaner = higher
        self.total_score   = 0.0
        self.elapsed_s     = 0.0
        self.timed_out     = False
        self.error         = None

    def compute_total(self, weights=(0.5, 0.3, 0.2)):
        w_logic, w_speed, w_mem = weights
        self.total_score = (
            w_logic * self.logic_score
            + w_speed * self.speed_score
            + w_mem  * self.memory_score
        )
        return self.total_score

    def __repr__(self):
        return (f"BenchmarkResult({self.mutation_id}: "
                f"logic={self.logic_score:.2f}, speed={self.speed_score:.2f}, "
                f"mem={self.memory_score:.2f}, total={self.total_score:.2f})")


class EvolutionArena:
    """
    The coordinator. Runs the full mutation → spawn → benchmark → select cycle.
    """

    def __init__(
        self,
        seed_path: str = "./jellyfish_seed/core_seed.json",
        workspace_dir: str = ".",
        population: int = DEFAULT_POPULATION,
        dry_run: bool = True,
    ):
        self.seed_path     = seed_path
        self.workspace_dir = workspace_dir
        self.population    = population
        self.dry_run       = dry_run
        self.mutator       = MutationEngine()
        self.generation_log: list[dict] = []

        mode = "🔵 DRY RUN" if dry_run else "🔴 LIVE"
        print(f"🧬 [EvolutionArena] 초기화 | {mode} | population={population}")

    # ── Main Entry ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the full evolutionary cycle.
        Returns the winning variant's full config dict.
        """
        seed = self._load_seed()
        generation = seed.get("generation", 0) + 1

        print(f"\n⚔️  ════════════════════════════════════════════════════════")
        print(f"⚔️   진화 아레나 가동 — 제{generation}세대 자연도태 시작")
        print(f"⚔️   개체 수: {self.population}  |  씨앗: {self.seed_path}")
        print(f"⚔️  ════════════════════════════════════════════════════════\n")

        bus.emit("evolution_started", {"generation": generation, "population": self.population})

        # Step 1: Mutate
        print(f"🔬 STEP 1/3: 변이체 생성 ({self.population}개)...")
        variants = self.mutator.mutate(seed, n=self.population)
        for v in variants:
            print(f"   🧬 {v['mutation_id']} → params={v['mutation_params']}")

        # Step 2: Benchmark race
        print(f"\n🏟  STEP 2/3: 벤치마크 가동 (제한시간: {BENCHMARK_TIMEOUT_S}s)...")
        results = self._run_benchmarks(variants)
        for r in sorted(results, key=lambda x: x.total_score, reverse=True):
            print(f"   {'🥇' if r == results[0] else '💀'} {r.mutation_id}: "
                  f"총점={r.total_score:.3f} | {r.elapsed_s:.1f}s")

        # Step 3: Select winner
        print(f"\n👑 STEP 3/3: 생존자 선택...")
        winner = max(results, key=lambda r: r.total_score)
        winner_variant = next(v for v in variants if v["mutation_id"] == winner.mutation_id)

        winner_variant["generation"] = generation
        winner_variant["winning_score"] = winner.total_score
        winner_variant["survivors_beat"] = [r.mutation_id for r in results if r.mutation_id != winner.mutation_id]

        # Save the winning variant as the new canonical seed
        if not self.dry_run:
            with open(self.seed_path, "w") as f:
                json.dump(winner_variant, f, indent=2, ensure_ascii=False)
            print(f"   💾 새 씨앗 저장: {self.seed_path}")

        bus.emit("evolution_complete", {
            "generation": generation,
            "winner_id": winner.mutation_id,
            "winner_score": winner.total_score,
        })

        print(f"\n👑 우승자: {winner.mutation_id} (총점: {winner.total_score:.3f})")
        print(f"   새 하이퍼파라미터: {winner_variant['mutation_params']}")
        self.generation_log.append({
            "generation": generation,
            "winner_id": winner.mutation_id,
            "score": winner.total_score,
            "params": winner_variant["mutation_params"],
        })

        if self.dry_run:
            print(f"\n🔵 DRY RUN: 씨앗 파일 변경 생략")

        return winner_variant

    # ── Benchmarking ───────────────────────────────────────────────────────────

    def _run_benchmarks(self, variants: list[dict]) -> list[BenchmarkResult]:
        """
        Run each variant through the benchmark gauntlet.
        In dry_run mode: run the built-in Python benchmarks directly.
        In live mode: spawn isolated subprocesses per variant.
        """
        results = []
        threads = []

        for variant in variants:
            result = BenchmarkResult(variant["mutation_id"])
            results.append(result)

            if self.dry_run:
                # Run inline for speed (no subprocess overhead in test mode)
                self._run_inline_benchmark(variant, result)
            else:
                t = threading.Thread(
                    target=self._run_subprocess_benchmark,
                    args=(variant, result),
                    daemon=True,
                )
                threads.append(t)
                t.start()

        for t in threads:
            t.join(timeout=BENCHMARK_TIMEOUT_S + 5)

        # Compute normalized total scores
        max_speed = max((r.elapsed_s for r in results if r.elapsed_s > 0), default=1)
        for r in results:
            r.speed_score = 1.0 - (r.elapsed_s / max_speed) if not r.timed_out else 0.0
            r.compute_total()

        return results

    def _run_inline_benchmark(self, variant: dict, result: BenchmarkResult):
        """
        Built-in benchmark gauntlet. Runs directly in this process.
        Used in dry_run mode to avoid subprocess overhead during testing.
        """
        t0 = time.time()
        params = variant.get("mutation_params", {})
        temperature = params.get("temperature", 0.7)
        ewc_lambda  = params.get("ewc_lambda", 1.0)

        try:
            # ── Test 1: Logic Puzzle Score ─────────────────────────────────
            # Simulate: "higher temperature = more creative but less accurate"
            logic_penalty = max(0, (temperature - 0.7) * 2)
            result.logic_score = max(0.0, 1.0 - logic_penalty + random.uniform(-0.05, 0.05))

            # ── Test 2: Memory Efficiency ──────────────────────────────────
            # Simulate: EWC lambda controls how aggressively old weights are preserved.
            # Too high = rigid, too low = forgets. Optimal is ~1.5
            ewc_ideal = 1.5
            ewc_delta = abs(ewc_lambda - ewc_ideal) / ewc_ideal
            result.memory_score = max(0.0, 1.0 - ewc_delta + random.uniform(-0.05, 0.05))

            # ── Simulate processing time ───────────────────────────────────
            # Slightly faster models get a speed bonus in the final scoring
            time.sleep(random.uniform(0.01, 0.1))

        except Exception as e:
            result.error = str(e)
            result.logic_score = 0.0
            result.memory_score = 0.0

        result.elapsed_s = time.time() - t0

    def _run_subprocess_benchmark(self, variant: dict, result: BenchmarkResult):
        """
        Spawns an isolated subprocess running `fza_arena_benchmark.py`,
        passing the variant config via environment variable.
        """
        t0 = time.time()
        env = os.environ.copy()
        env["FZA_ARENA_CONFIG"] = json.dumps(variant)

        try:
            proc = subprocess.run(
                ["python", BENCHMARK_SCRIPT],
                capture_output=True,
                text=True,
                timeout=BENCHMARK_TIMEOUT_S,
                env=env,
                cwd=self.workspace_dir,
            )
            if proc.returncode == 0:
                scores = json.loads(proc.stdout.strip().split("\n")[-1])
                result.logic_score  = scores.get("logic", 0.0)
                result.memory_score = scores.get("memory", 0.0)
            else:
                result.error = proc.stderr[:200]
        except subprocess.TimeoutExpired:
            result.timed_out = True
        except Exception as e:
            result.error = str(e)

        result.elapsed_s = time.time() - t0

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _load_seed(self) -> dict:
        if not os.path.exists(self.seed_path):
            # If no seed yet, use a synthetic one for first evolution
            print(f"⚠️  [EvolutionArena] 씨앗 없음. 기본 씨앗 사용.")
            return {
                "generation": 0,
                "root_axioms": [],
                "master_lora": {"preserved_skills": []},
            }
        with open(self.seed_path) as f:
            return json.load(f)

    def print_lineage(self):
        """Print evolutionary history."""
        if not self.generation_log:
            print("⚔️  [EvolutionArena] 아직 진화 기록 없음")
            return
        print(f"\n⚔️  [EvolutionArena] 진화 계보:")
        for g in self.generation_log:
            print(f"   Gen {g['generation']:3d}: {g['winner_id']} | "
                  f"점수={g['score']:.3f} | {g['params']}")
