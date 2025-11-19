DEFAULT_MODEL_PATH = "pretrained_ann.h5"
DEFAULT_SCALER_PATH = "scaler.joblib"
SAMPLE_CSV = "augmented_data_full.csv"
FIXED_DISPLACEMENT = 100.0  # µm
RNG_SEED = 42


POP_SIZE = 50
N_GENERATIONS = 50
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 3

# Bounds for Size (Area) and Pitch (µm)
BOUNDS = {
    "Area": (100.0, 10000.0),
    "Pitch": (50.0, 1000.0)
}
