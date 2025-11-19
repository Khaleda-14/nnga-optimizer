import numpy as np
from .constants import FIXED_DISPLACEMENT

class GeneticOptimizer:
    def __init__(self, model, scaler, fixed_displacement=FIXED_DISPLACEMENT,
                 pop_size=50, n_generations=50,
                 mutation_rate=0.15, tournament_size=3,
                 bounds=None, signals=None, rng=None):
        self.model = model
        self.scaler = scaler
        self.fixed_displacement = fixed_displacement
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.bounds = bounds or {"Area": (100.0,10000.0), "Pitch": (50.0,300.0)}
        self.signals = signals
        self.rng = rng or np.random.RandomState()

        self.dim = len(self.bounds)
        self.lb = np.array([self.bounds[k][0] for k in self.bounds], dtype=float)
        self.ub = np.array([self.bounds[k][1] for k in self.bounds], dtype=float)

    def _initialize_population(self):
        return self.rng.rand(self.pop_size, self.dim)

    def _decode(self, individual):
        return self.lb + individual * (self.ub - self.lb)

    def _predict(self, Area, pitch):
        inp = np.array([[self.fixed_displacement, float(Area), float(pitch)]], dtype=float)
        inp_scaled = self.scaler.transform(inp)
        pred = float(self.model.predict(inp_scaled, verbose=0)[0][0])
        return pred

    def _evaluate(self, individual):
        Area, pitch = self._decode(individual)
        pred = self._predict(Area, pitch)
        if np.isnan(pred) or pred <= 0 or pred < 0.01:
            return 1e6
        return pred

    def _tournament_select(self, population, fitness):
        selected = []
        for _ in range(self.pop_size):
            idxs = self.rng.choice(self.pop_size, size=self.tournament_size, replace=False)
            best = idxs[np.argmin(fitness[idxs])]
            selected.append(population[best].copy())
        return np.array(selected)

    def _crossover(self, p1, p2):
        alpha = self.rng.rand()
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        return c1, c2

    def _mutate(self, individual):
        if self.rng.rand() < self.mutation_rate:
            noise = self.rng.normal(0, 0.08, size=individual.shape)
            individual = individual + noise
            individual = np.clip(individual, 0.0, 1.0)
        return individual

    def run(self, init_individual=None):
        pop = self._initialize_population()
        if init_individual is not None:
            normalized_init = (init_individual - self.lb) / (self.ub - self.lb)
            normalized_init = np.clip(normalized_init, 0.0, 1.0)
            pop[0] = normalized_init

        best_history, mean_history = [], []
        best_individual, best_fitness = None, float('inf')

        for gen in range(self.n_generations):
            fitness = np.array([self._evaluate(ind) for ind in pop])
            gen_best_idx = np.argmin(fitness)
            gen_best_fit = fitness[gen_best_idx]
            gen_mean_fit = float(np.mean(fitness[np.isfinite(fitness)]))
            best_history.append(gen_best_fit)
            mean_history.append(gen_mean_fit)

            if gen_best_fit < best_fitness:
                best_fitness = gen_best_fit
                best_individual = pop[gen_best_idx].copy()

            if self.signals:
                percent = int(100 * (gen+1) / self.n_generations)
                self.signals.progress.emit(percent)
                self.signals.plot_update.emit({
                    "best_history": best_history.copy(),
                    "mean_history": mean_history.copy(),
                    "generation": gen+1
                })
                self.signals.log.emit(f"Gen {gen+1}/{self.n_generations}  Best = {gen_best_fit:.6f}")

            selected = self._tournament_select(pop, fitness)
            next_pop = []
            for i in range(0, self.pop_size, 2):
                p1 = selected[i]
                p2 = selected[(i+1) % self.pop_size]
                c1, c2 = self._crossover(p1, p2)
                next_pop.append(self._mutate(c1))
                next_pop.append(self._mutate(c2))
            pop = np.array(next_pop)[:self.pop_size]

        best_scaled = self._decode(best_individual)
        best_pred = self._predict(best_scaled[0], best_scaled[1])
        result = {
            "best_individual": best_scaled,
            "best_pred": best_pred,
            "best_history": best_history,
            "mean_history": mean_history
        }
        if self.signals:
            self.signals.log.emit("finished.")
            self.signals.finished.emit(result)
        return result
