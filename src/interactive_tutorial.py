"""
Interactive Tutorial System for DDR5 AI Sandbox Simulator
Provides step-by-step guidance for users learning DDR5 optimization.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import streamlit as st

from src.ddr5_models import DDR5Configuration, DDR5TimingParameters, DDR5VoltageParameters
from configuration_templates import ConfigurationTemplateManager, UseCase


class TutorialStep(Enum):
    """Tutorial step types."""
    INTRODUCTION = "introduction"
    EXPLANATION = "explanation"
    DEMONSTRATION = "demonstration"
    INTERACTION = "interaction"
    QUIZ = "quiz"
    SUMMARY = "summary"


@dataclass
class TutorialStepData:
    """Tutorial step data structure."""
    step_id: str
    step_type: TutorialStep
    title: str
    content: str
    objectives: List[str]
    prerequisites: List[str]
    estimated_time: int  # minutes
    difficulty: str  # beginner, intermediate, advanced
    interactive_elements: Dict[str, Any]
    validation_function: Optional[Callable]
    hints: List[str]
    further_reading: List[str]


class InteractiveTutorial:
    """Interactive tutorial system for DDR5 optimization."""
    
    def __init__(self):
        """Initialize the tutorial system."""
        self.tutorials: Dict[str, List[TutorialStepData]] = {}
        self.current_tutorial: Optional[str] = None
        self.current_step: int = 0
        self.user_progress: Dict[str, Dict[str, Any]] = {}
        self.template_manager = ConfigurationTemplateManager()
        
        # Load tutorials
        self._load_tutorials()
    
    def _load_tutorials(self):
        """Load all tutorial content."""
        self._load_beginner_tutorial()
        self._load_intermediate_tutorial()
        self._load_advanced_tutorial()
        self._load_gaming_tutorial()
        self._load_productivity_tutorial()
    
    def _load_beginner_tutorial(self):
        """Load beginner tutorial: DDR5 Basics."""
        tutorial_steps = [
            TutorialStepData(
                step_id="intro_welcome",
                step_type=TutorialStep.INTRODUCTION,
                title="Welcome to DDR5 Optimization!",
                content="""
                Welcome to the DDR5 AI Sandbox Simulator! This tutorial will teach you the 
                fundamentals of DDR5 memory optimization using artificial intelligence.
                
                **What you'll learn:**
                - Understanding DDR5 memory basics
                - Key timing parameters and their effects
                - Voltage configuration and safety
                - Using AI for automatic optimization
                - Interpreting performance results
                
                **Prerequisites:** Basic computer knowledge
                **Estimated time:** 30 minutes
                """,
                objectives=[
                    "Understand DDR5 memory fundamentals",
                    "Learn key timing parameters",
                    "Master voltage configuration",
                    "Use AI optimization effectively"
                ],
                prerequisites=[],
                estimated_time=30,
                difficulty="beginner",
                interactive_elements={},
                validation_function=None,
                hints=[],
                further_reading=[
                    "DDR5 JEDEC Standard Specification",
                    "Memory Overclocking Guide for Beginners"
                ]
            ),
            
            TutorialStepData(
                step_id="ddr5_basics",
                step_type=TutorialStep.EXPLANATION,
                title="DDR5 Memory Fundamentals",
                content="""
                **DDR5 (Double Data Rate 5)** is the latest generation of system memory, 
                offering significant improvements over DDR4:
                
                **Key Improvements:**
                - Higher data rates (3200-8400+ MT/s)
                - Improved power efficiency
                - Enhanced reliability features
                - Better signal integrity
                
                **Basic Terminology:**
                - **Frequency**: How fast the memory operates (MHz)
                - **Timings**: Delay cycles for memory operations
                - **Voltage**: Electrical power supplied to memory
                - **Bandwidth**: Data transfer capacity (GB/s)
                - **Latency**: Time delay for memory access (ns)
                
                **Memory Hierarchy:**
                1. CPU Registers (fastest, smallest)
                2. CPU Cache (L1, L2, L3)
                3. System Memory (DDR5) ← We optimize this!
                4. Storage (SSD/HDD, slowest, largest)
                """,
                objectives=[
                    "Understand DDR5 improvements over DDR4",
                    "Learn memory terminology",
                    "Understand memory hierarchy"
                ],
                prerequisites=["intro_welcome"],
                estimated_time=5,
                difficulty="beginner",
                interactive_elements={
                    "comparison_chart": True,
                    "terminology_quiz": True
                },
                validation_function=None,
                hints=[
                    "Think of memory like a highway - frequency is speed limit, timings are traffic lights",
                    "Higher frequency = more data transfer, but may need looser timings"
                ],
                further_reading=[
                    "DDR5 vs DDR4: Technical Comparison",
                    "Understanding Memory Hierarchy"
                ]
            ),
            
            TutorialStepData(
                step_id="timing_parameters",
                step_type=TutorialStep.EXPLANATION,
                title="Understanding Timing Parameters",
                content="""
                **Timing parameters** control how memory operations are scheduled. 
                Think of them as traffic lights for memory access:
                
                **Primary Timings:**
                - **CL (CAS Latency)**: Delay between column address and data output
                - **tRCD**: Row to Column Delay - time to open a row
                - **tRP**: Row Precharge - time to close a row
                - **tRAS**: Row Active Time - minimum time row must stay open
                - **tRC**: Row Cycle - complete row access time
                
                **Timing Relationships:**
                - tRAS ≥ tRCD + CL (row must stay open long enough)
                - tRC ≥ tRAS + tRP (complete cycle includes close time)
                
                **Performance Impact:**
                - Lower timings = better performance
                - Too aggressive = instability
                - Must balance with frequency and voltage
                
                **Common Notation:**
                DDR5-5600 CL36-38-38-76 means:
                - 5600 MT/s frequency
                - CL=36, tRCD=38, tRP=38, tRAS=76
                """,
                objectives=[
                    "Understand primary timing parameters",
                    "Learn timing relationships",
                    "Recognize timing notation"
                ],
                prerequisites=["ddr5_basics"],
                estimated_time=8,
                difficulty="beginner",
                interactive_elements={
                    "timing_calculator": True,
                    "relationship_validator": True
                },
                validation_function=lambda params: self._validate_timing_relationships(params),
                hints=[
                    "Lower timings = faster, but stability may suffer",
                    "tRAS must be at least tRCD + CL",
                    "Start with looser timings and tighten gradually"
                ],
                further_reading=[
                    "DDR5 Timing Parameters Deep Dive",
                    "Memory Timing Optimization Guide"
                ]
            ),
            
            TutorialStepData(
                step_id="voltage_config",
                step_type=TutorialStep.EXPLANATION,
                title="Voltage Configuration and Safety",
                content="""
                **Voltage** provides the electrical power for memory operation. 
                DDR5 uses multiple voltage rails:
                
                **DDR5 Voltage Rails:**
                - **VDDQ**: Main memory voltage (1.1V nominal)
                - **VPP**: Wordline voltage (1.8V nominal)
                - **VDD1**: Management voltage (1.8V nominal)
                - **VDD2**: Auxiliary voltage (1.1V nominal)
                - **VDDQ_TX**: Transmit voltage (1.1V nominal)
                
                **Voltage Effects:**
                - Higher voltage = better stability at high frequencies
                - Higher voltage = more power consumption and heat
                - Too high voltage = permanent damage
                
                **Safety Ranges:**
                - VDDQ: 1.0V - 1.4V (1.35V daily max recommended)
                - VPP: 1.7V - 2.0V (1.95V daily max recommended)
                
                **Overclocking Strategy:**
                1. Start with stock voltages
                2. Increase frequency first
                3. Add voltage only if needed for stability
                4. Monitor temperatures constantly
                """,
                objectives=[
                    "Understand DDR5 voltage rails",
                    "Learn voltage safety limits",
                    "Master voltage tuning strategy"
                ],
                prerequisites=["timing_parameters"],
                estimated_time=6,
                difficulty="beginner",
                interactive_elements={
                    "voltage_calculator": True,
                    "safety_checker": True
                },
                validation_function=lambda voltages: self._validate_voltage_safety(voltages),
                hints=[
                    "Start conservative - voltage damage is permanent",
                    "Monitor temperatures when increasing voltage",
                    "Small voltage changes can have big stability effects"
                ],
                further_reading=[
                    "DDR5 Voltage Rails Explained",
                    "Safe Overclocking Practices"
                ]
            ),
            
            TutorialStepData(
                step_id="ai_optimization",
                step_type=TutorialStep.DEMONSTRATION,
                title="AI-Powered Optimization",
                content="""
                **AI Optimization** uses machine learning to find optimal DDR5 settings:
                
                **How AI Helps:**
                - Tries thousands of configurations automatically
                - Learns from successful/failed attempts
                - Balances multiple objectives (speed, stability, power)
                - Finds settings humans might miss
                
                **AI Optimization Process:**
                1. **Training**: AI learns from database of known configurations
                2. **Evolution**: Genetic algorithm explores parameter space
                3. **Prediction**: ML models predict performance/stability
                4. **Refinement**: Iterative improvement of best candidates
                
                **Optimization Goals:**
                - **Balanced**: Best overall performance and stability
                - **Performance**: Maximum speed (may sacrifice stability)
                - **Stability**: Maximum reliability (may sacrifice speed)
                - **Efficiency**: Best performance per watt
                
                **Let's try it!** We'll optimize a DDR5-5600 configuration.
                """,
                objectives=[
                    "Understand AI optimization process",
                    "Learn different optimization goals",
                    "Run first AI optimization"
                ],
                prerequisites=["voltage_config"],
                estimated_time=10,
                difficulty="beginner",
                interactive_elements={
                    "ai_demo": True,
                    "goal_selector": True,
                    "optimization_runner": True
                },
                validation_function=lambda result: self._validate_ai_result(result),
                hints=[
                    "Start with 'balanced' goal for first optimization",
                    "AI may take 2-3 minutes to complete",
                    "Watch the evolution progress in real-time"
                ],
                further_reading=[
                    "Machine Learning in Memory Optimization",
                    "Genetic Algorithms for Hardware Tuning"
                ]
            ),
            
            TutorialStepData(
                step_id="results_interpretation",
                step_type=TutorialStep.EXPLANATION,
                title="Interpreting Results",
                content="""
                **Understanding optimization results** is crucial for successful tuning:
                
                **Key Metrics:**
                - **Bandwidth**: Data transfer rate (higher = better)
                - **Latency**: Access delay (lower = better)
                - **Stability**: Reliability score (higher = better)
                - **Power**: Energy consumption (lower = better)
                
                **Performance Indicators:**
                - **Bandwidth > 80,000 MB/s**: Good for DDR5-5600
                - **Latency < 70ns**: Excellent responsiveness
                - **Stability > 85%**: Suitable for daily use
                - **Power < 5000mW**: Reasonable consumption
                
                **Red Flags:**
                - Stability < 70%: May cause crashes
                - Excessive voltage: Risk of damage
                - High temperatures: Thermal throttling
                
                **Validation Steps:**
                1. Check AI confidence score
                2. Verify timing relationships
                3. Confirm voltage safety
                4. Review stability prediction
                """,
                objectives=[
                    "Understand performance metrics",
                    "Recognize good vs poor results",
                    "Learn validation process"
                ],
                prerequisites=["ai_optimization"],
                estimated_time=5,
                difficulty="beginner",
                interactive_elements={
                    "results_analyzer": True,
                    "metric_interpreter": True
                },
                validation_function=None,
                hints=[
                    "High bandwidth doesn't matter if stability is poor",
                    "Balance all metrics for best daily experience",
                    "AI confidence tells you how sure it is"
                ],
                further_reading=[
                    "Memory Performance Benchmarking",
                    "Stability Testing Best Practices"
                ]
            ),
            
            TutorialStepData(
                step_id="tutorial_summary",
                step_type=TutorialStep.SUMMARY,
                title="Tutorial Summary",
                content="""
                **Congratulations!** You've completed the DDR5 Basics tutorial!
                
                **What you've learned:**
                ✅ DDR5 memory fundamentals
                ✅ Timing parameters and relationships
                ✅ Voltage configuration and safety
                ✅ AI optimization process
                ✅ Results interpretation
                
                **Key Takeaways:**
                - Start conservative, increase gradually
                - Stability is more important than peak performance
                - AI can find configurations you might miss
                - Always validate results before applying
                
                **Next Steps:**
                - Try different optimization goals
                - Experiment with manual tuning
                - Take the Intermediate tutorial
                - Explore gaming-specific optimization
                
                **Quick Reference:**
                - Safe VDDQ range: 1.0V - 1.35V
                - Safe VPP range: 1.7V - 1.95V
                - Minimum stability: 70% for testing, 85% for daily use
                """,
                objectives=[
                    "Review key concepts",
                    "Plan next learning steps",
                    "Access quick reference"
                ],
                prerequisites=["results_interpretation"],
                estimated_time=3,
                difficulty="beginner",
                interactive_elements={
                    "knowledge_check": True,
                    "next_steps": True
                },
                validation_function=None,
                hints=[],
                further_reading=[
                    "Intermediate DDR5 Optimization Tutorial",
                    "Gaming Performance Optimization Guide"
                ]
            )
        ]
        
        self.tutorials["beginner_basics"] = tutorial_steps
    
    def _load_intermediate_tutorial(self):
        """Load intermediate tutorial: Advanced Optimization."""
        # Placeholder for intermediate tutorial
        self.tutorials["intermediate_advanced"] = []
    
    def _load_advanced_tutorial(self):
        """Load advanced tutorial: Expert Techniques."""
        # Placeholder for advanced tutorial
        self.tutorials["advanced_expert"] = []
    
    def _load_gaming_tutorial(self):
        """Load gaming-specific tutorial."""
        # Placeholder for gaming tutorial
        self.tutorials["gaming_optimization"] = []
    
    def _load_productivity_tutorial(self):
        """Load productivity-specific tutorial."""
        # Placeholder for productivity tutorial
        self.tutorials["productivity_optimization"] = []
    
    def get_available_tutorials(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available tutorials."""
        tutorial_info = {
            "beginner_basics": {
                "title": "DDR5 Basics",
                "description": "Learn DDR5 fundamentals and AI optimization",
                "difficulty": "beginner",
                "duration": 30,
                "steps": len(self.tutorials.get("beginner_basics", [])),
                "prerequisites": []
            },
            "intermediate_advanced": {
                "title": "Advanced Optimization",
                "description": "Manual tuning and advanced techniques",
                "difficulty": "intermediate",
                "duration": 45,
                "steps": len(self.tutorials.get("intermediate_advanced", [])),
                "prerequisites": ["beginner_basics"]
            },
            "advanced_expert": {
                "title": "Expert Techniques",
                "description": "Professional overclocking and custom profiles",
                "difficulty": "advanced",
                "duration": 60,
                "steps": len(self.tutorials.get("advanced_expert", [])),
                "prerequisites": ["intermediate_advanced"]
            },
            "gaming_optimization": {
                "title": "Gaming Optimization",
                "description": "Optimize DDR5 for gaming performance",
                "difficulty": "intermediate",
                "duration": 35,
                "steps": len(self.tutorials.get("gaming_optimization", [])),
                "prerequisites": ["beginner_basics"]
            },
            "productivity_optimization": {
                "title": "Productivity Optimization",
                "description": "Optimize for work and content creation",
                "difficulty": "intermediate",
                "duration": 40,
                "steps": len(self.tutorials.get("productivity_optimization", [])),
                "prerequisites": ["beginner_basics"]
            }
        }
        return tutorial_info
    
    def start_tutorial(self, tutorial_id: str) -> bool:
        """Start a specific tutorial."""
        if tutorial_id not in self.tutorials:
            return False
        
        self.current_tutorial = tutorial_id
        self.current_step = 0
        
        # Initialize progress tracking
        if tutorial_id not in self.user_progress:
            self.user_progress[tutorial_id] = {
                "started": True,
                "completed": False,
                "current_step": 0,
                "completed_steps": [],
                "quiz_scores": {},
                "time_spent": 0
            }
        
        return True
    
    def get_current_step(self) -> Optional[TutorialStepData]:
        """Get the current tutorial step."""
        if not self.current_tutorial or self.current_tutorial not in self.tutorials:
            return None
        
        steps = self.tutorials[self.current_tutorial]
        if self.current_step >= len(steps):
            return None
        
        return steps[self.current_step]
    
    def next_step(self) -> bool:
        """Move to the next tutorial step."""
        if not self.current_tutorial:
            return False
        
        steps = self.tutorials[self.current_tutorial]
        if self.current_step < len(steps) - 1:
            # Mark current step as completed
            self.user_progress[self.current_tutorial]["completed_steps"].append(self.current_step)
            self.current_step += 1
            self.user_progress[self.current_tutorial]["current_step"] = self.current_step
            return True
        else:
            # Tutorial completed
            self.user_progress[self.current_tutorial]["completed"] = True
            return False
    
    def previous_step(self) -> bool:
        """Move to the previous tutorial step."""
        if not self.current_tutorial or self.current_step <= 0:
            return False
        
        self.current_step -= 1
        self.user_progress[self.current_tutorial]["current_step"] = self.current_step
        return True
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current tutorial progress."""
        if not self.current_tutorial:
            return {}
        
        progress = self.user_progress.get(self.current_tutorial, {})
        total_steps = len(self.tutorials[self.current_tutorial])
        
        return {
            "tutorial_id": self.current_tutorial,
            "current_step": self.current_step,
            "total_steps": total_steps,
            "progress_percentage": (self.current_step / total_steps) * 100,
            "completed_steps": progress.get("completed_steps", []),
            "is_completed": progress.get("completed", False)
        }
    
    def render_tutorial_step(self, step: TutorialStepData):
        """Render a tutorial step in Streamlit."""
        # Display step header
        st.markdown(f"## {step.title}")
        
        # Progress indicator
        progress = self.get_progress()
        st.progress(progress["progress_percentage"] / 100)
        st.write(f"Step {progress['current_step'] + 1} of {progress['total_steps']}")
        
        # Step content
        st.markdown(step.content)
        
        # Objectives
        if step.objectives:
            st.markdown("### 📋 Objectives")
            for objective in step.objectives:
                st.write(f"• {objective}")
        
        # Interactive elements
        self._render_interactive_elements(step)
        
        # Hints
        if step.hints:
            with st.expander("💡 Hints"):
                for hint in step.hints:
                    st.write(f"• {hint}")
        
        # Navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("⬅️ Previous") and self.current_step > 0:
                self.previous_step()
                st.rerun()
        
        with col2:
            if step.step_type == TutorialStep.QUIZ:
                if st.button("✅ Submit Quiz"):
                    # Handle quiz submission
                    pass
            elif step.validation_function:
                if st.button("✅ Validate"):
                    # Handle validation
                    pass
        
        with col3:
            if st.button("Next ➡️"):
                if self.next_step():
                    st.rerun()
                else:
                    st.success("🎉 Tutorial completed!")
    
    def _render_interactive_elements(self, step: TutorialStepData):
        """Render interactive elements for a tutorial step."""
        elements = step.interactive_elements
        
        if elements.get("comparison_chart"):
            self._render_ddr5_comparison_chart()
        
        if elements.get("terminology_quiz"):
            self._render_terminology_quiz()
        
        if elements.get("timing_calculator"):
            self._render_timing_calculator()
        
        if elements.get("voltage_calculator"):
            self._render_voltage_calculator()
        
        if elements.get("ai_demo"):
            self._render_ai_demo()
    
    def _render_ddr5_comparison_chart(self):
        """Render DDR5 vs DDR4 comparison chart."""
        st.markdown("### 📊 DDR5 vs DDR4 Comparison")
        
        import pandas as pd
        
        comparison_data = {
            "Feature": ["Data Rate", "Voltage", "Channels", "Capacity", "Efficiency"],
            "DDR4": ["2133-4800 MT/s", "1.2V", "2", "Up to 32GB", "Good"],
            "DDR5": ["3200-8400+ MT/s", "1.1V", "2 (dual sub-channel)", "Up to 128GB", "Excellent"]
        }
        
        df = pd.DataFrame(comparison_data)
        st.table(df)
    
    def _render_terminology_quiz(self):
        """Render terminology quiz."""
        st.markdown("### 🎯 Quick Quiz: DDR5 Terminology")
        
        quiz_answers = {}
        
        quiz_answers["q1"] = st.radio(
            "What does 'MT/s' stand for?",
            ["Megabytes per second", "Megatransfers per second", "Megahertz per second"]
        )
        
        quiz_answers["q2"] = st.radio(
            "Which is faster?",
            ["Lower latency", "Higher latency", "They're the same"]
        )
        
        quiz_answers["q3"] = st.radio(
            "DDR5's nominal voltage is:",
            ["1.2V", "1.1V", "1.35V"]
        )
        
        if st.button("Check Answers"):
            correct = 0
            total = 3
            
            if quiz_answers["q1"] == "Megatransfers per second":
                correct += 1
            if quiz_answers["q2"] == "Lower latency":
                correct += 1
            if quiz_answers["q3"] == "1.1V":
                correct += 1
            
            score = (correct / total) * 100
            st.write(f"Score: {correct}/{total} ({score:.1f}%)")
            
            if score >= 80:
                st.success("Great job! 🎉")
            elif score >= 60:
                st.warning("Good effort! Review the material and try again.")
            else:
                st.error("Need more practice. Please review the tutorial content.")
    
    def _render_timing_calculator(self):
        """Render timing parameter calculator."""
        st.markdown("### 🧮 Timing Calculator")
        
        with st.form("timing_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                cl = st.number_input("CL (CAS Latency)", min_value=10, max_value=50, value=32)
                trcd = st.number_input("tRCD", min_value=10, max_value=50, value=32)
                trp = st.number_input("tRP", min_value=10, max_value=50, value=32)
            
            with col2:
                tras = st.number_input("tRAS", min_value=20, max_value=120, value=64)
                trc = st.number_input("tRC", min_value=40, max_value=200, value=96)
            
            if st.form_submit_button("Validate Timings"):
                violations = self._validate_timing_relationships({
                    'cl': cl, 'trcd': trcd, 'trp': trp, 'tras': tras, 'trc': trc
                })
                
                if not violations:
                    st.success("✅ All timing relationships are valid!")
                else:
                    st.error("❌ Timing violations:")
                    for violation in violations:
                        st.write(f"• {violation}")
    
    def _render_voltage_calculator(self):
        """Render voltage safety calculator."""
        st.markdown("### ⚡ Voltage Safety Calculator")
        
        with st.form("voltage_form"):
            vddq = st.slider("VDDQ", 1.0, 1.4, 1.1, 0.01)
            vpp = st.slider("VPP", 1.7, 2.0, 1.8, 0.01)
            
            if st.form_submit_button("Check Safety"):
                safety_result = self._validate_voltage_safety({'vddq': vddq, 'vpp': vpp})
                
                if safety_result["safe"]:
                    st.success("✅ Voltages are within safe ranges!")
                else:
                    st.error("❌ Voltage safety concerns:")
                    for concern in safety_result["concerns"]:
                        st.write(f"• {concern}")
    
    def _render_ai_demo(self):
        """Render AI optimization demo."""
        st.markdown("### 🤖 AI Optimization Demo")
        
        goal = st.selectbox(
            "Select optimization goal:",
            ["balanced", "performance", "stability", "efficiency"]
        )
        
        frequency = st.selectbox(
            "Target frequency:",
            [5600, 6000, 6400, 6800]
        )
        
        if st.button("🚀 Run AI Optimization"):
            with st.spinner("AI is optimizing your configuration..."):
                # Simulate AI optimization
                import time
                time.sleep(3)
                
                # Mock results
                result = {
                    "bandwidth": 85000 + (frequency - 5600) * 3,
                    "latency": 65 - (frequency - 5600) * 0.002,
                    "stability": 88 - (frequency - 5600) * 0.003,
                    "power": 3500 + (frequency - 5600) * 0.5,
                    "confidence": 85
                }
                
                st.success("✅ Optimization complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Bandwidth", f"{result['bandwidth']:,} MB/s")
                    st.metric("Latency", f"{result['latency']:.1f} ns")
                
                with col2:
                    st.metric("Stability", f"{result['stability']:.1f}%")
                    st.metric("Power", f"{result['power']:.0f} mW")
                
                st.info(f"AI Confidence: {result['confidence']}%")
    
    def _validate_timing_relationships(self, params: Dict[str, int]) -> List[str]:
        """Validate timing parameter relationships."""
        violations = []
        
        # tRAS >= tRCD + CL
        if params['tras'] < params['trcd'] + params['cl']:
            violations.append(f"tRAS ({params['tras']}) must be >= tRCD + CL ({params['trcd'] + params['cl']})")
        
        # tRC >= tRAS + tRP
        if params['trc'] < params['tras'] + params['trp']:
            violations.append(f"tRC ({params['trc']}) must be >= tRAS + tRP ({params['tras'] + params['trp']})")
        
        return violations
    
    def _validate_voltage_safety(self, voltages: Dict[str, float]) -> Dict[str, Any]:
        """Validate voltage safety."""
        concerns = []
        
        if voltages['vddq'] > 1.35:
            concerns.append("VDDQ above daily safe limit (1.35V)")
        
        if voltages['vpp'] > 1.95:
            concerns.append("VPP above daily safe limit (1.95V)")
        
        return {
            "safe": len(concerns) == 0,
            "concerns": concerns
        }
    
    def _validate_ai_result(self, result: Dict[str, Any]) -> bool:
        """Validate AI optimization result."""
        return (
            result.get("bandwidth", 0) > 50000 and
            result.get("stability", 0) > 70 and
            result.get("confidence", 0) > 60
        )
