from src.ai_optimizer import GeneticAlgorithmOptimizer
from src.ddr5_models import DDR5Configuration


def test_ga_random_configs_obey_jedec_like_constraints():
    base = DDR5Configuration()
    ga = GeneticAlgorithmOptimizer(population_size=4, max_generations=1)

    cfg = ga.create_random_config(base)

    # Frequency bounds
    assert 3200 <= cfg.frequency <= 8400
    # Voltage bounds
    assert 1.05 <= cfg.voltages.vddq <= 1.25
    assert 1.7 <= cfg.voltages.vpp <= 2.0
    # Timing relationships
    assert cfg.timings.tras >= cfg.timings.trcd + cfg.timings.cl
    assert cfg.timings.trc >= cfg.timings.tras + cfg.timings.trp
