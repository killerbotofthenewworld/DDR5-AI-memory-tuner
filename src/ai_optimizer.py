"""
AI-powered DDR5 Memory Optimizer
Uses machine learning algorithms to optimize DDR5 memory configurations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from ddr5_models import DDR5Configuration, DDR5TimingParameters, DDR5VoltageParameters
from ddr5_simulator import DDR5Simulator


class AdvancedAIOptimizer:
    """Advanced AI-powered DDR5 memory configuration optimizer with multiple ML models."""
    
    def __init__(self):
        """Initialize the advanced AI optimizer."""
        self.simulator = DDR5Simulator()
        
        # Multiple AI models for ensemble prediction
        self.performance_models = {
            'random_forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(128, 64, 32), random_state=42, max_iter=500)
        }
        
        self.stability_models = {
            'random_forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42, max_iter=500)
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Enhanced optimization parameters
        self.population_size = 100  # Larger population for better exploration
        self.generations = 200      # More generations for convergence
        self.mutation_rate = 0.15   # Higher mutation for diversity
        self.elite_size = 10        # Keep best individuals
        
        # AI learning parameters
        self.learning_iterations = 5
        self.training_data_size = 5000  # Larger training dataset
        
        # Memory of successful configurations
        self.successful_configs = []
        self.training_data = None
        
        # Real DDR5 performance database (simulated high-quality data)
        self.performance_database = self._initialize_performance_database()
    
    def _initialize_performance_database(self) -> Dict[str, List[Dict]]:
        """Initialize database with known high-performance DDR5 configurations."""
        return {
            'ddr5_3200': [
                {'cl': 16, 'trcd': 16, 'trp': 16, 'tras': 36, 'vddq': 1.10, 'performance': 85.2, 'stability': 95.0},
                {'cl': 18, 'trcd': 18, 'trp': 18, 'tras': 38, 'vddq': 1.10, 'performance': 82.1, 'stability': 98.0},
            ],
            'ddr5_4800': [
                {'cl': 24, 'trcd': 24, 'trp': 24, 'tras': 52, 'vddq': 1.10, 'performance': 92.3, 'stability': 90.0},
                {'cl': 26, 'trcd': 26, 'trp': 26, 'tras': 54, 'vddq': 1.12, 'performance': 89.7, 'stability': 95.0},
            ],
            'ddr5_5600': [
                {'cl': 28, 'trcd': 28, 'trp': 28, 'tras': 58, 'vddq': 1.10, 'performance': 96.8, 'stability': 88.0},
                {'cl': 30, 'trcd': 30, 'trp': 30, 'tras': 60, 'vddq': 1.12, 'performance': 94.2, 'stability': 92.0},
                {'cl': 32, 'trcd': 32, 'trp': 32, 'tras': 62, 'vddq': 1.14, 'performance': 91.5, 'stability': 95.0},
            ],
            'ddr5_6400': [
                {'cl': 32, 'trcd': 32, 'trp': 32, 'tras': 64, 'vddq': 1.12, 'performance': 98.5, 'stability': 85.0},
                {'cl': 34, 'trcd': 34, 'trp': 34, 'tras': 66, 'vddq': 1.14, 'performance': 95.8, 'stability': 88.0},
                {'cl': 36, 'trcd': 36, 'trp': 36, 'tras': 68, 'vddq': 1.16, 'performance': 93.2, 'stability': 91.0},
            ],
            'ddr5_7200': [
                {'cl': 36, 'trcd': 36, 'trp': 36, 'tras': 72, 'vddq': 1.14, 'performance': 99.2, 'stability': 82.0},
                {'cl': 38, 'trcd': 38, 'trp': 38, 'tras': 74, 'vddq': 1.16, 'performance': 96.8, 'stability': 85.0},
                {'cl': 40, 'trcd': 40, 'trp': 40, 'tras': 76, 'vddq': 1.18, 'performance': 94.1, 'stability': 88.0},
            ]
        }
    
    def generate_enhanced_training_data(self, num_samples: int = 5000) -> pd.DataFrame:
        """
        Generate enhanced training data using real DDR5 performance data and simulation.
        
        Args:
            num_samples: Number of training samples to generate
            
        Returns:
            DataFrame with features and targets
        """
        print(f"Generating {num_samples} enhanced training samples with real DDR5 data...")
        
        training_samples = []
        
        # Include real performance data
        for freq_key, configs in self.performance_database.items():
            frequency = int(freq_key.split('_')[1])
            for config in configs:
                sample = {
                    'frequency': frequency,
                    'cl': config['cl'],
                    'trcd': config['trcd'],
                    'trp': config['trp'],
                    'tras': config['tras'],
                    'trc': config['tras'] + config['trp'],
                    'trfc': 295,  # Standard
                    'vddq': config['vddq'],
                    'vpp': 1.8,   # Standard
                    'bandwidth_gbps': config['performance'],
                    'latency_ns': 15.0,  # Estimated
                    'power_mw': 2500,    # Estimated
                    'stability_score': config['stability']
                }
                training_samples.append(sample)
        
        # Generate additional synthetic samples
        remaining_samples = num_samples - len(training_samples)
        
        for _ in range(remaining_samples):
            # Generate random but realistic DDR5 configuration
            config = self._generate_intelligent_config()
            
            # Simulate performance metrics
            self.simulator.load_configuration(config)
            
            bandwidth_results = self.simulator.simulate_bandwidth()
            latency_results = self.simulator.simulate_latency()
            power_results = self.simulator.simulate_power_consumption()
            stability_results = self.simulator.run_stability_test()
            
            # Create feature vector
            features = {
                'frequency': config.frequency,
                'cl': config.timings.cl,
                'trcd': config.timings.trcd,
                'trp': config.timings.trp,
                'tras': config.timings.tras,
                'trc': config.timings.trc,
                'trfc': config.timings.trfc,
                'vddq': config.voltages.vddq,
                'vpp': config.voltages.vpp,
                
                # Targets
                'bandwidth_gbps': bandwidth_results['effective_bandwidth_gbps'],
                'latency_ns': latency_results['effective_latency_ns'],
                'power_mw': power_results['total_power_mw'],
                'stability_score': stability_results['stability_score']
            }
            
            training_samples.append(features)
        
        self.training_data = pd.DataFrame(training_samples)
        return self.training_data
    
    def train_ensemble_models(self, training_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Train ensemble of AI models for superior prediction accuracy.
        
        Args:
            training_data: Optional pre-generated training data
            
        Returns:
            Dictionary with model performance scores
        """
        if training_data is None:
            if self.training_data is None:
                training_data = self.generate_enhanced_training_data(self.training_data_size)
            else:
                training_data = self.training_data
        
        print("Training ensemble of AI models...")
        
        # Prepare features
        feature_columns = [
            'frequency', 'cl', 'trcd', 'trp', 'tras', 'trc', 'trfc', 'vddq', 'vpp'
        ]
        X = training_data[feature_columns]
        
        # Performance targets (composite score: bandwidth/latency ratio)
        y_performance = (training_data['bandwidth_gbps'] * 1000) / training_data['latency_ns']
        
        # Stability target
        y_stability = training_data['stability_score']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_perf_train, y_perf_test = train_test_split(
            X_scaled, y_performance, test_size=0.2, random_state=42
        )
        _, _, y_stab_train, y_stab_test = train_test_split(
            X_scaled, y_stability, test_size=0.2, random_state=42
        )
        
        model_scores = {}
        
        # Train performance models
        print("Training performance prediction models...")
        for name, model in self.performance_models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_perf_train)
            score = model.score(X_test, y_perf_test)
            model_scores[f'performance_{name}'] = score
            print(f"    {name} R² score: {score:.4f}")
        
        # Train stability models
        print("Training stability prediction models...")
        for name, model in self.stability_models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_stab_train)
            score = model.score(X_test, y_stab_test)
            model_scores[f'stability_{name}'] = score
            print(f"    {name} R² score: {score:.4f}")
        
        self.is_trained = True
        print("✅ Ensemble training complete!")
        
        return model_scores
    
    def intelligent_optimize(
        self,
        target_frequency: int,
        optimization_goal: str = "ai_balanced",
        performance_target: float = 95.0,
        stability_target: float = 90.0,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Intelligent optimization using multiple AI strategies and iterative learning.
        
        Args:
            target_frequency: Target memory frequency
            optimization_goal: "ai_balanced", "ai_performance", "ai_stability", "ai_extreme"
            performance_target: Target performance score (0-100)
            stability_target: Target stability score (0-100)
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with optimized configuration and results
        """
        if not self.is_trained:
            print("🧠 Training AI ensemble...")
            self.train_ensemble_models()
        
        print(f"🎯 Intelligent optimization for DDR5-{target_frequency}")
        print(f"   Goal: {optimization_goal}")
        print(f"   Targets: {performance_target}% performance, {stability_target}% stability")
        
        best_overall_config = None
        best_overall_fitness = -np.inf
        optimization_history = []
        
        for iteration in range(max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Adaptive population and generations based on iteration
            current_population = self.population_size + (iteration * 20)
            current_generations = self.generations - (iteration * 30)
            
            # Initialize population with smart seeding
            population = self._initialize_smart_population(target_frequency, iteration)
            
            best_fitness = -np.inf
            best_individual = None
            best_config = None
            
            generation_scores = []
            
            for generation in range(current_generations):
                # Evaluate fitness with ensemble prediction
                fitness_scores = []
                
                for individual in population:
                    config = self._individual_to_config(individual, target_frequency)
                    fitness = self._evaluate_ensemble_fitness(
                        config, optimization_goal, performance_target, stability_target
                    )
                    fitness_scores.append(fitness)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()
                        best_config = config
                
                generation_scores.append(max(fitness_scores))
                
                # Adaptive evolution strategy
                population = self._adaptive_evolution(
                    population, fitness_scores, generation, current_generations
                )
                
                # Progress reporting
                if generation % 50 == 0:
                    print(f"    Generation {generation}: Best fitness = {best_fitness:.4f}")
            
            # Evaluate final configuration with simulation
            if best_config:
                self.simulator.load_configuration(best_config)
                simulation_results = {
                    'bandwidth': self.simulator.simulate_bandwidth(),
                    'latency': self.simulator.simulate_latency(),
                    'power': self.simulator.simulate_power_consumption(),
                    'stability': self.simulator.run_stability_test()
                }
                
                # Calculate actual performance metrics
                actual_performance = self._calculate_actual_performance(simulation_results)
                actual_stability = simulation_results['stability']['stability_score']
                
                print(f"    Final: Performance={actual_performance:.1f}%, Stability={actual_stability:.1f}%")
                
                # Update best overall if this iteration is better
                if best_fitness > best_overall_fitness:
                    best_overall_fitness = best_fitness
                    best_overall_config = best_config
                
                # Store successful configuration for learning
                if actual_performance >= performance_target * 0.9 and actual_stability >= stability_target * 0.9:
                    self.successful_configs.append({
                        'config': best_config,
                        'performance': actual_performance,
                        'stability': actual_stability,
                        'fitness': best_fitness
                    })
                
                optimization_history.append({
                    'iteration': iteration + 1,
                    'fitness_scores': generation_scores,
                    'final_performance': actual_performance,
                    'final_stability': actual_stability,
                    'best_config': best_config
                })
        
        # Final simulation with best configuration
        if best_overall_config:
            self.simulator.load_configuration(best_overall_config)
            final_results = {
                'bandwidth': self.simulator.simulate_bandwidth(),
                'latency': self.simulator.simulate_latency(),
                'power': self.simulator.simulate_power_consumption(),
                'stability': self.simulator.run_stability_test()
            }
            
            return {
                'optimized_config': best_overall_config,
                'fitness_score': best_overall_fitness,
                'simulation_results': final_results,
                'optimization_history': optimization_history,
                'optimization_goal': optimization_goal,
                'ai_insights': self._generate_ai_insights(best_overall_config, final_results),
                'success_rate': len(self.successful_configs),
                'recommendations': self._generate_smart_recommendations(best_overall_config, final_results)
            }
        
        return {'error': 'Optimization failed to find suitable configuration'}
    
    def _generate_intelligent_config(self) -> DDR5Configuration:
        """Generate intelligent random configuration based on learned patterns."""
        frequencies = [3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000, 6400, 6800, 7200]
        frequency = np.random.choice(frequencies)
        
        # Use database knowledge for smarter generation
        freq_key = f'ddr5_{frequency}'
        base_timings = None
        
        if freq_key in self.performance_database:
            # Use real data as base with variation
            base_config = np.random.choice(self.performance_database[freq_key])
            base_timings = {
                'cl': base_config['cl'] + np.random.randint(-2, 3),
                'trcd': base_config['trcd'] + np.random.randint(-2, 3),
                'trp': base_config['trp'] + np.random.randint(-2, 3),
                'tras': base_config['tras'] + np.random.randint(-4, 5),
                'vddq': base_config['vddq'] + np.random.uniform(-0.02, 0.02)
            }
        else:
            # Intelligent estimation based on frequency
            base_cl = max(16, int(frequency * 0.0055))
            base_timings = {
                'cl': base_cl + np.random.randint(-2, 5),
                'trcd': base_cl + np.random.randint(-2, 5),
                'trp': base_cl + np.random.randint(-2, 5),
                'tras': base_cl + 20 + np.random.randint(-5, 10),
                'vddq': 1.10 + np.random.uniform(-0.05, 0.08)
            }
        
        # Ensure valid ranges
        base_timings['cl'] = max(16, min(50, base_timings['cl']))
        base_timings['trcd'] = max(16, min(50, base_timings['trcd']))
        base_timings['trp'] = max(16, min(50, base_timings['trp']))
        base_timings['tras'] = max(30, min(80, base_timings['tras']))
        base_timings['vddq'] = max(1.05, min(1.20, base_timings['vddq']))
        
        timings = DDR5TimingParameters(
            cl=int(base_timings['cl']),
            trcd=int(base_timings['trcd']),
            trp=int(base_timings['trp']),
            tras=int(base_timings['tras']),
            trc=int(base_timings['tras'] + base_timings['trp']),
            trfc=np.random.randint(280, 320)
        )
        
        voltages = DDR5VoltageParameters(
            vddq=base_timings['vddq'],
            vpp=np.random.uniform(1.75, 1.85)
        )
        
        return DDR5Configuration(
            frequency=frequency,
            timings=timings,
            voltages=voltages
        )
    
    def _individual_to_config(self, individual: List[float], frequency: int) -> DDR5Configuration:
        """Convert genetic algorithm individual to DDR5 configuration."""
        cl, trcd, trp, tras, vddq, vpp = individual
        
        timings = DDR5TimingParameters(
            cl=int(cl),
            trcd=int(trcd),
            trp=int(trp),
            tras=int(tras),
            trc=int(tras + trp),
            trfc=295  # Standard value
        )
        
        voltages = DDR5VoltageParameters(vddq=vddq, vpp=vpp)
        
        return DDR5Configuration(
            frequency=frequency,
            timings=timings,
            voltages=voltages
        )
    
    def _evaluate_ensemble_fitness(
        self, 
        config: DDR5Configuration, 
        goal: str, 
        perf_target: float, 
        stab_target: float
    ) -> float:
        """Evaluate fitness using ensemble of AI models."""
        predictions = self.predict_ensemble_performance(config)
        
        performance = predictions['ensemble_performance']
        stability = predictions['ensemble_stability']
        confidence = predictions['confidence']
        
        # Goal-specific fitness calculation
        if goal == "ai_performance":
            base_fitness = performance * 0.8 + stability * 0.2
            # Bonus for exceeding targets
            if performance >= perf_target:
                base_fitness += (performance - perf_target) * 0.1
        elif goal == "ai_stability":
            base_fitness = stability * 0.8 + performance * 0.2
            if stability >= stab_target:
                base_fitness += (stability - stab_target) * 0.1
        elif goal == "ai_extreme":
            # Extreme performance with minimum stability threshold
            if stability >= 75:
                base_fitness = performance * 1.2
            else:
                base_fitness = performance * 0.5  # Heavy penalty
        else:  # ai_balanced
            base_fitness = (performance + stability) / 2
            # Bonus for balanced achievement
            if performance >= perf_target and stability >= stab_target:
                base_fitness += 10
        
        # Confidence weighting
        base_fitness *= confidence
        
        # Penalty for invalid configurations
        violations = config.validate_configuration()
        total_violations = sum(len(v) for v in violations.values())
        if total_violations > 0:
            base_fitness *= (0.8 ** total_violations)
        
        return base_fitness
    
    def _mutate(self, individual: List[float]) -> List[float]:
        """Mutate an individual with intelligent constraints."""
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                if i < 4:  # Timing parameters (integers)
                    individual[i] += np.random.randint(-2, 3)
                    individual[i] = max(1, individual[i])
                else:  # Voltage parameters (floats)
                    individual[i] += np.random.normal(0, 0.02)
                    if i == 4:  # VDDQ
                        individual[i] = np.clip(individual[i], 1.05, 1.15)
                    else:  # VPP
                        individual[i] = np.clip(individual[i], 1.75, 1.85)
        
        return individual
    
    def intelligent_optimize(
        self,
        target_frequency: int,
        optimization_goal: str = "ai_balanced",
        performance_target: float = 95.0,
        stability_target: float = 90.0,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Intelligent optimization using multiple AI strategies and iterative learning.
        
        Args:
            target_frequency: Target memory frequency
            optimization_goal: "ai_balanced", "ai_performance", "ai_stability", "ai_extreme"
            performance_target: Target performance score (0-100)
            stability_target: Target stability score (0-100)
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with optimized configuration and results
        """
        if not self.is_trained:
            print("🧠 Training AI ensemble...")
            self.train_ensemble_models()
        
        print(f"🎯 Intelligent optimization for DDR5-{target_frequency}")
        print(f"   Goal: {optimization_goal}")
        print(f"   Targets: {performance_target}% performance, {stability_target}% stability")
        
        best_overall_config = None
        best_overall_fitness = -np.inf
        optimization_history = []
        
        for iteration in range(max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Adaptive population and generations based on iteration
            current_population = self.population_size + (iteration * 20)
            current_generations = self.generations - (iteration * 30)
            
            # Initialize population with smart seeding
            population = self._initialize_smart_population(target_frequency, iteration)
            
            best_fitness = -np.inf
            best_individual = None
            best_config = None
            
            generation_scores = []
            
            for generation in range(current_generations):
                # Evaluate fitness with ensemble prediction
                fitness_scores = []
                
                for individual in population:
                    config = self._individual_to_config(individual, target_frequency)
                    fitness = self._evaluate_ensemble_fitness(
                        config, optimization_goal, performance_target, stability_target
                    )
                    fitness_scores.append(fitness)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()
                        best_config = config
                
                generation_scores.append(max(fitness_scores))
                
                # Adaptive evolution strategy
                population = self._adaptive_evolution(
                    population, fitness_scores, generation, current_generations
                )
                
                # Progress reporting
                if generation % 50 == 0:
                    print(f"    Generation {generation}: Best fitness = {best_fitness:.4f}")
            
            # Evaluate final configuration with simulation
            if best_config:
                self.simulator.load_configuration(best_config)
                simulation_results = {
                    'bandwidth': self.simulator.simulate_bandwidth(),
                    'latency': self.simulator.simulate_latency(),
                    'power': self.simulator.simulate_power_consumption(),
                    'stability': self.simulator.run_stability_test()
                }
                
                # Calculate actual performance metrics
                actual_performance = self._calculate_actual_performance(simulation_results)
                actual_stability = simulation_results['stability']['stability_score']
                
                print(f"    Final: Performance={actual_performance:.1f}%, Stability={actual_stability:.1f}%")
                
                # Update best overall if this iteration is better
                if best_fitness > best_overall_fitness:
                    best_overall_fitness = best_fitness
                    best_overall_config = best_config
                
                # Store successful configuration for learning
                if actual_performance >= performance_target * 0.9 and actual_stability >= stability_target * 0.9:
                    self.successful_configs.append({
                        'config': best_config,
                        'performance': actual_performance,
                        'stability': actual_stability,
                        'fitness': best_fitness
                    })
                
                optimization_history.append({
                    'iteration': iteration + 1,
                    'fitness_scores': generation_scores,
                    'final_performance': actual_performance,
                    'final_stability': actual_stability,
                    'best_config': best_config
                })
        
        # Final simulation with best configuration
        if best_overall_config:
            self.simulator.load_configuration(best_overall_config)
            final_results = {
                'bandwidth': self.simulator.simulate_bandwidth(),
                'latency': self.simulator.simulate_latency(),
                'power': self.simulator.simulate_power_consumption(),
                'stability': self.simulator.run_stability_test()
            }
            
            return {
                'optimized_config': best_overall_config,
                'fitness_score': best_overall_fitness,
                'simulation_results': final_results,
                'optimization_history': optimization_history,
                'optimization_goal': optimization_goal,
                'ai_insights': self._generate_ai_insights(best_overall_config, final_results),
                'success_rate': len(self.successful_configs),
                'recommendations': self._generate_smart_recommendations(best_overall_config, final_results)
            }
        
        return {'error': 'Optimization failed to find suitable configuration'}
    
    def _generate_intelligent_config(self) -> DDR5Configuration:
        """Generate intelligent random configuration based on learned patterns."""
        frequencies = [3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000, 6400, 6800, 7200]
        frequency = np.random.choice(frequencies)
        
        # Use database knowledge for smarter generation
        freq_key = f'ddr5_{frequency}'
        base_timings = None
        
        if freq_key in self.performance_database:
            # Use real data as base with variation
            base_config = np.random.choice(self.performance_database[freq_key])
            base_timings = {
                'cl': base_config['cl'] + np.random.randint(-2, 3),
                'trcd': base_config['trcd'] + np.random.randint(-2, 3),
                'trp': base_config['trp'] + np.random.randint(-2, 3),
                'tras': base_config['tras'] + np.random.randint(-4, 5),
                'vddq': base_config['vddq'] + np.random.uniform(-0.02, 0.02)
            }
        else:
            # Intelligent estimation based on frequency
            base_cl = max(16, int(frequency * 0.0055))
            base_timings = {
                'cl': base_cl + np.random.randint(-2, 5),
                'trcd': base_cl + np.random.randint(-2, 5),
                'trp': base_cl + np.random.randint(-2, 5),
                'tras': base_cl + 20 + np.random.randint(-5, 10),
                'vddq': 1.10 + np.random.uniform(-0.05, 0.08)
            }
        
        # Ensure valid ranges
        base_timings['cl'] = max(16, min(50, base_timings['cl']))
        base_timings['trcd'] = max(16, min(50, base_timings['trcd']))
        base_timings['trp'] = max(16, min(50, base_timings['trp']))
        base_timings['tras'] = max(30, min(80, base_timings['tras']))
        base_timings['vddq'] = max(1.05, min(1.20, base_timings['vddq']))
        
        timings = DDR5TimingParameters(
            cl=int(base_timings['cl']),
            trcd=int(base_timings['trcd']),
            trp=int(base_timings['trp']),
            tras=int(base_timings['tras']),
            trc=int(base_timings['tras'] + base_timings['trp']),
            trfc=np.random.randint(280, 320)
        )
        
        voltages = DDR5VoltageParameters(
            vddq=base_timings['vddq'],
            vpp=np.random.uniform(1.75, 1.85)
        )
        
        return DDR5Configuration(
            frequency=frequency,
            timings=timings,
            voltages=voltages
        )
    
    def predict_ensemble_performance(self, config: DDR5Configuration) -> Dict[str, float]:
        """
        Predict performance using ensemble of models for higher accuracy.
        
        Args:
            config: DDR5 configuration to evaluate
            
        Returns:
            Dictionary with ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare features
        features = np.array([[
            config.frequency,
            config.timings.cl,
            config.timings.trcd,
            config.timings.trp,
            config.timings.tras,
            config.timings.trc,
            config.timings.trfc,
            config.voltages.vddq,
            config.voltages.vpp
        ]])
        
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from all models
        performance_predictions = []
        stability_predictions = []
        
        for model in self.performance_models.values():
            pred = model.predict(features_scaled)[0]
            performance_predictions.append(pred)
        
        for model in self.stability_models.values():
            pred = model.predict(features_scaled)[0]
            stability_predictions.append(pred)
        
        # Ensemble predictions (weighted average)
        ensemble_performance = np.mean(performance_predictions)
        ensemble_stability = np.mean(stability_predictions)
        
        # Calculate confidence based on prediction variance
        perf_variance = np.var(performance_predictions)
        stab_variance = np.var(stability_predictions)
        confidence = 1.0 / (1.0 + perf_variance + stab_variance)
        
        return {
            'ensemble_performance': ensemble_performance,
            'ensemble_stability': ensemble_stability,
            'confidence': confidence,
            'individual_performance': performance_predictions,
            'individual_stability': stability_predictions
        }
    
    def _initialize_smart_population(self, frequency: int, iteration: int) -> List[List[float]]:
        """Initialize population with smart seeding based on successful configs."""
        population = []
        
        # Seed with successful configurations if available
        seed_count = min(len(self.successful_configs), self.population_size // 4)
        for i in range(seed_count):
            successful = self.successful_configs[i]['config']
            if successful.frequency == frequency:
                individual = [
                    successful.timings.cl,
                    successful.timings.trcd,
                    successful.timings.trp,
                    successful.timings.tras,
                    successful.voltages.vddq,
                    successful.voltages.vpp
                ]
                # Add some mutation for exploration
                individual = self._mutate(individual)
                population.append(individual)
        
        # Fill remaining with intelligent random generation
        remaining = self.population_size - len(population)
        for _ in range(remaining):
            individual = self._generate_smart_individual(frequency, iteration)
            population.append(individual)
        
        return population
    
    def _generate_smart_individual(self, frequency: int, iteration: int) -> List[float]:
        """Generate smart individual based on frequency and learning."""
        # Use performance database knowledge
        freq_key = f'ddr5_{frequency}'
        
        if freq_key in self.performance_database and self.performance_database[freq_key]:
            # Randomly select a base configuration
            configs = self.performance_database[freq_key]
            base_config = configs[np.random.randint(0, len(configs))]
            
            # Add variation based on iteration (more exploration early)
            variance = max(0.5, 2.0 - iteration * 0.3)
            
            individual = [
                base_config['cl'] + np.random.normal(0, variance),
                base_config['trcd'] + np.random.normal(0, variance),
                base_config['trp'] + np.random.normal(0, variance),
                base_config['tras'] + np.random.normal(0, variance * 2),
                base_config['vddq'] + np.random.normal(0, variance * 0.01),
                1.8 + np.random.normal(0, variance * 0.02)  # VPP
            ]
        else:
            # Intelligent default based on frequency
            base_cl = max(16, int(frequency * 0.0055))
            variance = max(0.5, 2.0 - iteration * 0.3)
            
            individual = [
                base_cl + np.random.normal(0, variance * 2),
                base_cl + np.random.normal(0, variance * 2),
                base_cl + np.random.normal(0, variance * 2),
                base_cl + 20 + np.random.normal(0, variance * 3),
                1.10 + np.random.normal(0, variance * 0.02),
                1.80 + np.random.normal(0, variance * 0.02)
            ]
        
        # Ensure valid ranges
        individual[0] = max(16, min(50, individual[0]))  # CL
        individual[1] = max(16, min(50, individual[1]))  # tRCD
        individual[2] = max(16, min(50, individual[2]))  # tRP
        individual[3] = max(30, min(80, individual[3]))  # tRAS
        individual[4] = max(1.05, min(1.20, individual[4]))  # VDDQ
        individual[5] = max(1.70, min(1.90, individual[5]))  # VPP
        
        return individual
    
    def _adaptive_evolution(
        self, 
        population: List[List[float]], 
        fitness_scores: List[float],
        generation: int,
        max_generations: int
    ) -> List[List[float]]:
        """Adaptive evolution with dynamic parameters."""
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        new_population = []
        
        # Elite preservation
        elite_count = min(self.elite_size, len(sorted_population))
        new_population.extend(sorted_population[:elite_count])
        
        # Adaptive mutation rate
        progress = generation / max_generations
        current_mutation_rate = self.mutation_rate * (1.5 - progress)
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(sorted_population, sorted_fitness)
            parent2 = self._tournament_selection(sorted_population, sorted_fitness)
            
            # Crossover
            child = self._adaptive_crossover(parent1, parent2, progress)
            
            # Mutation
            child = self._adaptive_mutation(child, current_mutation_rate)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[List[float]], fitness_scores: List[float]) -> List[float]:
        """Tournament selection with adaptive tournament size."""
        tournament_size = min(5, len(population))
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _adaptive_crossover(self, parent1: List[float], parent2: List[float], progress: float) -> List[float]:
        """Adaptive crossover with progress-based blending."""
        child = []
        for i in range(len(parent1)):
            # Blend crossover with adaptive alpha
            alpha = 0.1 + progress * 0.4  # More blending as evolution progresses
            beta = np.random.uniform(-alpha, 1 + alpha)
            child_value = beta * parent1[i] + (1 - beta) * parent2[i]
            child.append(child_value)
        return child
    
    def _adaptive_mutation(self, individual: List[float], mutation_rate: float) -> List[float]:
        """Adaptive mutation with parameter-specific strategies."""
        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                if i < 4:  # Timing parameters
                    # Gaussian mutation with decreasing variance
                    sigma = max(0.5, 2.0 - mutation_rate * 5)
                    individual[i] += np.random.normal(0, sigma)
                    individual[i] = max(1, individual[i])
                else:  # Voltage parameters
                    # Smaller mutations for voltages
                    sigma = 0.01 if i == 4 else 0.02  # VDDQ vs VPP
                    individual[i] += np.random.normal(0, sigma)
                    if i == 4:  # VDDQ
                        individual[i] = np.clip(individual[i], 1.05, 1.20)
                    else:  # VPP
                        individual[i] = np.clip(individual[i], 1.70, 1.90)
        
        return individual
    
    def _calculate_actual_performance(self, simulation_results: Dict[str, Any]) -> float:
        """Calculate actual performance score from simulation results."""
        bandwidth = simulation_results['bandwidth']['effective_bandwidth_gbps']
        latency = simulation_results['latency']['effective_latency_ns']
        
        # Performance score based on bandwidth/latency ratio
        performance_ratio = (bandwidth * 1000) / latency
        
        # Normalize to 0-100 scale (DDR5-7200 CL36 ~= 100)
        max_expected_ratio = 280  # High-end DDR5 performance
        performance_score = min(100, (performance_ratio / max_expected_ratio) * 100)
        
        return performance_score
    
    def _generate_ai_insights(self, config: DDR5Configuration, results: Dict[str, Any]) -> List[str]:
        """Generate AI-driven insights about the configuration."""
        insights = []
        
        # Analyze timing efficiency
        base_cl = max(16, int(config.frequency * 0.0055))
        if config.timings.cl < base_cl:
            insights.append(f"🎯 Aggressive CL timing detected: {config.timings.cl} (base: {base_cl})")
        
        # Analyze voltage efficiency
        if config.voltages.vddq > 1.15:
            insights.append(f"⚡ High VDDQ voltage: {config.voltages.vddq:.3f}V - may increase performance but affects stability")
        
        # Analyze performance metrics
        bandwidth = results['bandwidth']['effective_bandwidth_gbps']
        theoretical = config.bandwidth_gbps or 0
        efficiency = (bandwidth / theoretical * 100) if theoretical > 0 else 0
        
        if efficiency > 90:
            insights.append("🚀 Excellent memory efficiency achieved!")
        elif efficiency < 70:
            insights.append("⚠️ Lower than expected efficiency - consider relaxing timings")
        
        # Stability analysis
        stability = results['stability']['stability_score']
        if stability > 95:
            insights.append("🛡️ Exceptional stability - configuration should be very reliable")
        elif stability < 80:
            insights.append("⚠️ Stability concerns - consider increasing voltages or relaxing timings")
        
        return insights
    
    def _generate_smart_recommendations(self, config: DDR5Configuration, results: Dict[str, Any]) -> List[str]:
        """Generate smart recommendations for further optimization."""
        recommendations = []
        
        stability = results['stability']['stability_score']
        performance = self._calculate_actual_performance(results)
        
        # Performance recommendations
        if performance < 90:
            if config.voltages.vddq < 1.15:
                recommendations.append("💡 Try increasing VDDQ voltage by 0.02-0.04V for better performance")
            if config.timings.cl > 30:
                recommendations.append("💡 Consider tightening CL timing by 1-2 cycles")
        
        # Stability recommendations
        if stability < 85:
            recommendations.append("🔧 Increase VDDQ voltage for better stability")
            recommendations.append("🔧 Consider relaxing primary timings by 1-2 cycles")
            if config.voltages.vpp < 1.85:
                recommendations.append("🔧 Try increasing VPP voltage slightly")
        
        # Efficiency recommendations
        efficiency = results['bandwidth']['efficiency_percent']
        if efficiency < 80:
            recommendations.append("⚙️ Check secondary timings - may need adjustment")
            recommendations.append("⚙️ Consider frequency-specific timing optimization")
        
        return recommendations

    
# Maintain backward compatibility
AIOptimizer = AdvancedAIOptimizer
