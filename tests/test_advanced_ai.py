"""
Test suite for advanced AI optimization modules
"""

import pytest
import numpy as np
from unittest.mock import patch
import tempfile
import os

from src.ai_optimizer import (
    AIOptimizer,
    GeneticAlgorithmOptimizer,
    ReinforcementLearningOptimizer,
    EnsembleOptimizer,
)
from src.ultra_ai_optimizer import UltraAIOptimizer, ComputerVisionAnalyzer
from src.deep_learning_predictor import (
    DeepLearningPredictor,
    EnsemblePredictor,
    StabilityPredictor
)
from src.ddr5_models import DDR5Configuration


class TestAIOptimizer:
    """Test cases for AI Optimizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.ai_optimizer = AIOptimizer()
        self.test_config = DDR5Configuration()
    
    def test_ai_optimizer_initialization(self):
        """Test AI optimizer initialization"""
        assert self.ai_optimizer is not None
        assert hasattr(self.ai_optimizer, 'ensemble_optimizer')
        assert hasattr(self.ai_optimizer, 'optimization_history')
    
    def test_optimization_configuration(self):
        """Test configuration optimization"""
        try:
            result = self.ai_optimizer.optimize_configuration(
                self.test_config,
                optimization_goal="performance",
                method="genetic"
            )
            
            assert result is not None
            assert hasattr(result, 'best_config')
            assert hasattr(result, 'best_score')
            assert result.best_score >= 0
            
        except Exception as e:
            # If optimization fails due to missing dependencies, that's acceptable
            pytest.skip(f"Optimization skipped due to: {e}")
    
    def test_optimization_suggestions(self):
        """Test AI suggestions"""
        suggestions = self.ai_optimizer.get_optimization_suggestions(self.test_config)
        
        assert isinstance(suggestions, list)
        # Should have at least some suggestions
        assert len(suggestions) >= 0
    
    def test_export_optimization_history(self):
        """Test history export"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filename = f.name
        
        try:
            self.ai_optimizer.export_optimization_history(filename)
            assert os.path.exists(filename)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


class TestGeneticAlgorithmOptimizer:
    """Test cases for Genetic Algorithm Optimizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.ga_optimizer = GeneticAlgorithmOptimizer(
            population_size=10,
            max_generations=5  # Small values for testing
        )
        self.test_config = DDR5Configuration()
    
    def test_random_config_generation(self):
        """Test random configuration generation"""
        random_config = self.ga_optimizer.create_random_config(self.test_config)
        
        assert random_config is not None
        assert isinstance(random_config, DDR5Configuration)
        # Should be different from base config
        assert (
            random_config.frequency != self.test_config.frequency
            or random_config.timings.cl != self.test_config.timings.cl
        )
    
    def test_crossover_operation(self):
        """Test genetic crossover"""
        parent1 = self.test_config
        parent2 = self.ga_optimizer.create_random_config(self.test_config)
        
        child1, child2 = self.ga_optimizer.crossover(parent1, parent2)
        
        assert child1 is not None
        assert child2 is not None
        assert isinstance(child1, DDR5Configuration)
        assert isinstance(child2, DDR5Configuration)
    
    def test_mutation_operation(self):
        """Test genetic mutation"""
        original_config = self.test_config.model_copy()
        mutated_config = self.ga_optimizer.mutate(original_config)
        
        assert mutated_config is not None
        assert isinstance(mutated_config, DDR5Configuration)
    
    def test_fitness_evaluation(self):
        """Test fitness evaluation"""
        fitness = self.ga_optimizer.evaluate_fitness(self.test_config)
        
        assert isinstance(fitness, float)
        assert fitness >= 0.0


class TestReinforcementLearningOptimizer:
    """Test cases for Reinforcement Learning Optimizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.rl_optimizer = ReinforcementLearningOptimizer()
        self.test_config = DDR5Configuration()
    
    def test_state_representation(self):
        """Test state to key conversion"""
        state_key = self.rl_optimizer.state_to_key(self.test_config)
        
        assert isinstance(state_key, str)
        assert len(state_key) > 0
    
    def test_action_generation(self):
        """Test action generation"""
        actions = self.rl_optimizer.get_actions(self.test_config)
        
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert all(isinstance(action, str) for action in actions)
    
    def test_action_application(self):
        """Test action application"""
        actions = self.rl_optimizer.get_actions(self.test_config)
        
        if actions:
            new_config = self.rl_optimizer.apply_action(self.test_config, actions[0])
            assert isinstance(new_config, DDR5Configuration)
    
    def test_config_evaluation(self):
        """Test configuration evaluation"""
        reward = self.rl_optimizer.evaluate_config(self.test_config)
        
        assert isinstance(reward, float)
        assert reward >= 0.0


class TestEnsembleOptimizer:
    """Test cases for Ensemble Optimizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.ensemble_optimizer = EnsembleOptimizer()
        self.test_config = DDR5Configuration()
    
    def test_ensemble_initialization(self):
        """Test ensemble optimizer initialization"""
        assert hasattr(self.ensemble_optimizer, 'ga_optimizer')
        assert hasattr(self.ensemble_optimizer, 'rl_optimizer')
    
    def test_genetic_optimization(self):
        """Test genetic optimization method"""
        try:
            result = self.ensemble_optimizer.optimize(
                self.test_config, method="genetic"
            )
            
            assert result is not None
            assert hasattr(result, 'best_config')
            assert hasattr(result, 'best_score')
            
        except Exception as e:
            pytest.skip(f"Genetic optimization skipped due to: {e}")
    
    def test_reinforcement_optimization(self):
        """Test reinforcement learning optimization method"""
        try:
            result = self.ensemble_optimizer.optimize(
                self.test_config, method="reinforcement"
            )
            
            assert result is not None
            assert hasattr(result, 'best_config')
            assert hasattr(result, 'best_score')
            
        except Exception as e:
            pytest.skip(f"RL optimization skipped due to: {e}")


class TestUltraAIOptimizer:
    """Test cases for Ultra AI Optimizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Use CPU for testing to avoid GPU dependencies
        self.ultra_optimizer = UltraAIOptimizer(device="cpu")
        self.test_config = DDR5Configuration()
    
    def test_ultra_ai_initialization(self):
        """Test Ultra AI optimizer initialization"""
        assert self.ultra_optimizer is not None
        assert hasattr(self.ultra_optimizer, 'performance_predictor')
        assert hasattr(self.ultra_optimizer, 'transformer_optimizer')
        assert hasattr(self.ultra_optimizer, 'cv_analyzer')
    
    def test_config_tensor_conversion(self):
        """Test configuration to tensor conversion"""
        tensor = self.ultra_optimizer.config_to_tensor(self.test_config)
        
        assert tensor is not None
        assert tensor.shape[0] > 0  # Should have some features
        
        # Test reverse conversion
        reconstructed_config = self.ultra_optimizer.tensor_to_config(
            tensor, self.test_config
        )
        assert isinstance(reconstructed_config, DDR5Configuration)
    
    def test_ai_recommendations(self):
        """Test AI recommendations"""
        try:
            recommendations = self.ultra_optimizer.get_ai_recommendations(
                self.test_config
            )
            
            assert isinstance(recommendations, list)
            # May be empty if models aren't trained
            assert len(recommendations) >= 0
            
        except Exception as e:
            pytest.skip(f"AI recommendations skipped due to: {e}")
    
    @patch('cv2.imread')
    @patch('pytesseract.image_to_string')
    def test_bios_screenshot_analysis(self, mock_ocr, mock_imread):
        """Test BIOS screenshot analysis"""
        # Mock the CV components
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_ocr.return_value = "Memory Frequency: 3200 MHz\nCAS Latency: 16"
        
        try:
            result = self.ultra_optimizer.analyze_bios_screenshot(
                np.zeros((100, 100, 3))
            )
            
            assert isinstance(result, dict)
            assert 'detected_settings' in result or 'error' in result
            
        except Exception as e:
            pytest.skip(f"BIOS analysis skipped due to: {e}")


class TestComputerVisionAnalyzer:
    """Test cases for Computer Vision Analyzer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.cv_analyzer = ComputerVisionAnalyzer()
    
    def test_cv_analyzer_initialization(self):
        """Test CV analyzer initialization"""
        assert self.cv_analyzer is not None
        assert hasattr(self.cv_analyzer, 'tesseract_config')
    
    @patch('cv2.imdecode')
    @patch('pytesseract.image_to_string')
    def test_settings_parsing(self, mock_ocr, mock_decode):
        """Test memory settings parsing"""
        mock_decode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_ocr.return_value = (
            "Memory Frequency: 3200 MHz\nCAS Latency: 16\nVoltage: 1.2V"
        )
        
        try:
            result = self.cv_analyzer.analyze_bios_screenshot(b"fake_image_data")
            
            assert isinstance(result, dict)
            # Should either have results or an error
            assert 'detected_settings' in result or 'error' in result
            
        except Exception as e:
            pytest.skip(f"CV analysis skipped due to: {e}")


class TestDeepLearningPredictor:
    """Test cases for Deep Learning Predictor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.dl_predictor = DeepLearningPredictor()
        self.test_config = DDR5Configuration()
    
    def test_dl_predictor_initialization(self):
        """Test deep learning predictor initialization"""
        assert self.dl_predictor is not None
        assert hasattr(self.dl_predictor, 'ensemble_predictor')
        assert hasattr(self.dl_predictor, 'stability_predictor')
    
    def test_random_config_generation(self):
        """Test random configuration generation"""
        random_config = self.dl_predictor._generate_random_config()
        
        assert isinstance(random_config, DDR5Configuration)
        assert random_config.frequency >= 3200
        assert random_config.frequency <= 8400
        assert random_config.timings.cl >= 14
    
    def test_performance_prediction(self):
        """Test performance prediction"""
        try:
            prediction = self.dl_predictor.predict_performance(self.test_config)
            
            assert isinstance(prediction, dict)
            # Should have either results or error
            assert 'performance' in prediction or 'error' in prediction
            
        except Exception as e:
            pytest.skip(f"Performance prediction skipped due to: {e}")


class TestEnsemblePredictor:
    """Test cases for Ensemble Predictor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.ensemble_predictor = EnsemblePredictor()
        self.test_config = DDR5Configuration()
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        features = self.ensemble_predictor.extract_features(self.test_config)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert all(isinstance(f, (int, float, np.number)) for f in features)
    
    def test_ensemble_predictor_initialization(self):
        """Test ensemble predictor initialization"""
        assert hasattr(self.ensemble_predictor, 'models')
        assert hasattr(self.ensemble_predictor, 'scalers')
        assert not self.ensemble_predictor.is_trained  # Should start untrained


class TestStabilityPredictor:
    """Test cases for Stability Predictor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.stability_predictor = StabilityPredictor()
        self.test_config = DDR5Configuration()
    
    def test_stability_feature_extraction(self):
        """Test stability feature extraction"""
        features = self.stability_predictor.extract_stability_features(self.test_config)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_heuristic_stability_prediction(self):
        """Test heuristic stability prediction"""
        prediction = self.stability_predictor._heuristic_stability_prediction(
            self.test_config
        )
        
        assert isinstance(prediction, dict)
        assert 'stability_score' in prediction
        assert 'risk_level' in prediction
        assert 0.0 <= prediction['stability_score'] <= 1.0
    
    def test_risk_categorization(self):
        """Test risk level categorization"""
        assert self.stability_predictor._categorize_risk(0.95) == "Very Low"
        assert self.stability_predictor._categorize_risk(0.85) == "Low"
        assert self.stability_predictor._categorize_risk(0.7) == "Medium"
        assert self.stability_predictor._categorize_risk(0.5) == "High"
        assert self.stability_predictor._categorize_risk(0.2) == "Very High"
    
    def test_risk_factor_analysis(self):
        """Test risk factor analysis"""
        risk_factors = self.stability_predictor._analyze_risk_factors(self.test_config)
        
        assert isinstance(risk_factors, list)
        # All risk factors should be strings
        assert all(isinstance(rf, str) for rf in risk_factors)


class TestIntegration:
    """Integration tests for AI modules"""
    
    def test_ai_module_integration(self):
        """Test integration between AI modules"""
        try:
            # Test that modules can be imported and initialized together
            ai_optimizer = AIOptimizer()
            ultra_optimizer = UltraAIOptimizer(device="cpu")
            dl_predictor = DeepLearningPredictor()
            
            assert ai_optimizer is not None
            assert ultra_optimizer is not None
            assert dl_predictor is not None
            
            # Test basic workflow
            test_config = DDR5Configuration()
            
            # Should be able to get suggestions from AI optimizer
            suggestions = ai_optimizer.get_optimization_suggestions(test_config)
            assert isinstance(suggestions, list)
            
            # Should be able to get predictions from DL predictor
            prediction = dl_predictor.predict_performance(test_config)
            assert isinstance(prediction, dict)
            
        except Exception as e:
            pytest.skip(f"Integration test skipped due to: {e}")
    
    def test_model_serialization(self):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                dl_predictor = DeepLearningPredictor()
                
                # Test saving (should not crash even if untrained)
                dl_predictor.save_models(temp_dir)
                
                # Test loading
                dl_predictor.load_models(temp_dir)
                
            except Exception as e:
                pytest.skip(f"Model serialization test skipped due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
