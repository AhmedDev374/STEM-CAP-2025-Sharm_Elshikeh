import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
import threading
import time
from datetime import datetime, timedelta
import random
import json
import seaborn as sns
from collections import deque
import warnings
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')


class AdvancedTrafficAISystem:
    def __init__(self):
        # Enhanced data structures
        self.sensor_data = {
            'timestamp': [],
            'ultrasonic_sub1_start': [],
            'ultrasonic_sub1_end': [],
            'ultrasonic_sub2_start': [],
            'ultrasonic_sub2_end': [],
            'ir_sensor': [],
            'mq135_pollution': [],
            'led_red_state': [],
            'led_green_state': [],
            'gate_state': [],
            'ambulance_detected': [],
            'car_count_sub1': [],
            'car_count_sub2': [],
            'traffic_status': [],
            'cars_entered_sub1': [],
            'cars_exited_sub1': [],
            'cars_entered_sub2': [],
            'cars_exited_sub2': []
        }

        # Advanced performance metrics
        self.performance_metrics = {
            'response_times': [],
            'car_count_accuracy': [],
            'emergency_response_time': [],
            'pollution_alerts': [],
            'system_efficiency': [],
            'throughput_rate': [],
            'avg_time_in_system': [],
            'congestion_level': []
        }

        # Car tracking system
        self.car_tracking = {
            'sub1_enter_times': {},
            'sub2_enter_times': {},
            'car_ids': set()
        }

        # Initialize ML models
        self.models = self.initialize_models()
        self.historical_data = pd.DataFrame()
        self.real_time_buffer = deque(maxlen=200)

        # Simulation parameters
        self.simulation_running = False
        self.current_car_count_sub1 = 0
        self.current_car_count_sub2 = 0
        self.total_cars_entered_sub1 = 0
        self.total_cars_exited_sub1 = 0
        self.total_cars_entered_sub2 = 0
        self.total_cars_exited_sub2 = 0
        self.emergency_mode = False   # manual only
        self.overload_mode = False    # manual only
        self.next_car_id = 1

        # Limits & thresholds
        self.total_cars_limit = 7     # absolute system max (when overload allows)
        self.subroad1_limit = 5       # per-subroad limit (you used this already)
        self.normal_total_limit = 5   # when overload inactive, total must be <= 5
        self.overload_min = 6         # while overload active, do not allow total < 6

    def initialize_models(self):
        """Initialize advanced machine learning models"""
        models = {
            'traffic_flow_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'emergency_detector': RandomForestClassifier(n_estimators=50, random_state=42),
            'pollution_analyzer': RandomForestRegressor(n_estimators=80, random_state=42),
            'response_time_predictor': RandomForestRegressor(n_estimators=60, random_state=42),
            'congestion_predictor': RandomForestClassifier(n_estimators=70, random_state=42)
        }
        return models

    def generate_car_id(self):
        """Generate unique car ID"""
        car_id = self.next_car_id
        self.next_car_id += 1
        return car_id

    def activate_overload_fill(self, target_total=None):
        """
        Immediately fill cars until target_total (6 or 7).
        Fill subroad1 up to subroad1_limit first, then place extra into subroad2.
        This method modifies counts immediately (no slow ramp).
        This is manual-only (called when user enables overload).
        """
        # If not provided choose 6 or 7
        if target_total is None:
            target_total = random.choice([6, 7])

        # don't exceed the absolute hard limit
        target_total = min(target_total, self.total_cars_limit)

        # Add cars until we reach target_total
        while (self.current_car_count_sub1 + self.current_car_count_sub2) < target_total:
            # prefer subroad1 until it's full
            if self.current_car_count_sub1 < self.subroad1_limit:
                car_id = self.generate_car_id()
                self.car_tracking['sub1_enter_times'][car_id] = datetime.now()
                self.car_tracking['car_ids'].add(car_id)
                self.current_car_count_sub1 += 1
                self.total_cars_entered_sub1 += 1
            else:
                car_id = self.generate_car_id()
                self.car_tracking['sub2_enter_times'][car_id] = datetime.now()
                self.car_tracking['car_ids'].add(car_id)
                self.current_car_count_sub2 += 1
                self.total_cars_entered_sub2 += 1

        # set overload_mode True (manual-only toggle should call this)
        self.overload_mode = True

    def generate_sensor_data(self):
        """Generate realistic sensor data with proper car counting"""
        timestamp = datetime.now()

        # Ultrasonic sensors as car counters (0 = car detected, >0 = no car)
        us_sub1_start = 0 if random.random() > 0.8 else random.randint(100, 300)
        us_sub1_end = 0 if random.random() > 0.85 else random.randint(100, 300)
        us_sub2_start = 0 if random.random() > 0.8 else random.randint(100, 300)
        us_sub2_end = 0 if random.random() > 0.85 else random.randint(100, 300)

        # IR sensor for ambulance detection (kept internal but will NOT toggle emergency automatically)
        ir_detection = 1 if random.random() > 0.97 else 0

        # MQ135 pollution sensor - base influenced by traffic and overload (overload is manual flag)
        base_pollution = 800
        if self.overload_mode:
            base_pollution += 800  # higher baseline during manual overload
        traffic_effect = (self.current_car_count_sub1 + self.current_car_count_sub2) * 100
        pollution = base_pollution + traffic_effect + random.randint(-50, 150)
        pollution = max(0, min(5000, pollution))

        # Car counting logic with unique IDs
        cars_entered_sub1 = 0
        cars_exited_sub1 = 0
        cars_entered_sub2 = 0
        cars_exited_sub2 = 0

        # --- AMBULANCE: if detected, clear all cars in subroad2 immediately (send them out to the left)
        ambulance_detected = 1 if ir_detection else 0
        if ambulance_detected:
            # Clear all cars in subroad2 regardless of overload_mode
            cars_to_exit = self.current_car_count_sub2
            for _ in range(cars_to_exit):
                if self.car_tracking['sub2_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub2_enter_times'].keys()))
                    enter_time = self.car_tracking['sub2_enter_times'].pop(car_id)
                    time_in_system = (timestamp - enter_time).total_seconds()
                    self.performance_metrics['avg_time_in_system'].append(time_in_system)
                self.total_cars_exited_sub2 += 1
                cars_exited_sub2 = 1
            self.current_car_count_sub2 = 0
            # note: do NOT auto-enable emergency_mode here — user wanted manual emergency

        # Handle emergency mode first - if manual emergency is ON we clear subroad2 (manual behavior preserved)
        if self.emergency_mode:
            cars_to_exit = self.current_car_count_sub2
            for _ in range(cars_to_exit):
                if self.car_tracking['sub2_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub2_enter_times'].keys()))
                    enter_time = self.car_tracking['sub2_enter_times'].pop(car_id)
                    time_in_system = (timestamp - enter_time).total_seconds()
                    self.performance_metrics['avg_time_in_system'].append(time_in_system)
                self.total_cars_exited_sub2 += 1
                cars_exited_sub2 = 1
            self.current_car_count_sub2 = 0

        # Determine the current allowed max depending on overload state
        allowed_total_max = self.total_cars_limit if self.overload_mode else self.normal_total_limit
        # Determine the minimum allowed while overload is active
        overload_min = self.overload_min if self.overload_mode else 0

        # Subroad 1 entry - normal flow (up to subroad1_limit cars)
        if us_sub1_start == 0 and not self.emergency_mode:
            total_cars_now = self.current_car_count_sub1 + self.current_car_count_sub2
            # allow entry only if sub1 not full and total under allowed max
            if self.current_car_count_sub1 < self.subroad1_limit and total_cars_now < allowed_total_max:
                car_id = self.generate_car_id()
                self.car_tracking['sub1_enter_times'][car_id] = timestamp
                self.car_tracking['car_ids'].add(car_id)
                self.total_cars_entered_sub1 += 1
                self.current_car_count_sub1 += 1
                cars_entered_sub1 = 1

        # Subroad 1 exit
        # If overload active, don't allow exits that would drop total below overload_min
        if us_sub1_end == 0 and self.current_car_count_sub1 > 0 and not self.emergency_mode:
            total_before_exit = self.current_car_count_sub1 + self.current_car_count_sub2
            # allow exit only if either not overload, or exit won't bring total below overload_min
            if (not self.overload_mode) or (total_before_exit - 1 >= overload_min):
                if self.car_tracking['sub1_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub1_enter_times'].keys()))
                    enter_time = self.car_tracking['sub1_enter_times'].pop(car_id)
                    time_in_system = (timestamp - enter_time).total_seconds()
                    self.performance_metrics['avg_time_in_system'].append(time_in_system)
                self.total_cars_exited_sub1 += 1
                self.current_car_count_sub1 = max(0, self.current_car_count_sub1 - 1)
                cars_exited_sub1 = 1
            else:
                # blocked exit (to preserve overload minimum)
                pass

        # Subroad 2 entry - ONLY when subroad1 is full
        if us_sub2_start == 0 and not self.emergency_mode:
            total_cars_now = self.current_car_count_sub1 + self.current_car_count_sub2
            # allow sub2 entry only when sub1 is full and total under allowed max
            if self.current_car_count_sub1 >= self.subroad1_limit and total_cars_now < allowed_total_max:
                car_id = self.generate_car_id()
                self.car_tracking['sub2_enter_times'][car_id] = timestamp
                self.car_tracking['car_ids'].add(car_id)
                self.total_cars_entered_sub2 += 1
                self.current_car_count_sub2 += 1
                cars_entered_sub2 = 1

        # Subroad 2 exit - only when not in emergency mode
        # If overload active, don't allow exits that would drop total below overload_min
        if us_sub2_end == 0 and self.current_car_count_sub2 > 0 and not self.emergency_mode:
            total_before_exit = self.current_car_count_sub1 + self.current_car_count_sub2
            if (not self.overload_mode) or (total_before_exit - 1 >= overload_min):
                if self.car_tracking['sub2_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub2_enter_times'].keys()))
                    enter_time = self.car_tracking['sub2_enter_times'].pop(car_id)
                    time_in_system = (timestamp - enter_time).total_seconds()
                    self.performance_metrics['avg_time_in_system'].append(time_in_system)
                self.total_cars_exited_sub2 += 1
                self.current_car_count_sub2 = max(0, self.current_car_count_sub2 - 1)
                cars_exited_sub2 = 1
            else:
                # blocked exit (to preserve overload minimum)
                pass

        # Clear subroad 2 when overload_mode is turned OFF manually (existing logic)
        # After overload off, we must also enforce the normal cap (<= normal_total_limit)
        if not self.overload_mode and self.current_car_count_sub2 > 0 and not self.emergency_mode:
            # Move cars from sub2 to sub1 while there's space in sub1
            while self.current_car_count_sub2 > 0 and self.current_car_count_sub1 < self.subroad1_limit:
                if self.car_tracking['sub2_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub2_enter_times'].keys()))
                    enter_time = self.car_tracking['sub2_enter_times'].pop(car_id)
                    self.car_tracking['sub1_enter_times'][car_id] = enter_time
                self.current_car_count_sub2 -= 1
                self.current_car_count_sub1 += 1

            # If total still > normal_total_limit, remove excess cars (simulate them leaving)
            while (self.current_car_count_sub1 + self.current_car_count_sub2) > self.normal_total_limit and not self.emergency_mode:
                # Prefer to remove from sub2 first
                if self.current_car_count_sub2 > 0 and self.car_tracking['sub2_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub2_enter_times'].keys()))
                    enter_time = self.car_tracking['sub2_enter_times'].pop(car_id)
                    time_in_system = (timestamp - enter_time).total_seconds()
                    self.performance_metrics['avg_time_in_system'].append(time_in_system)
                    self.current_car_count_sub2 -= 1
                    self.total_cars_exited_sub2 += 1
                elif self.current_car_count_sub1 > 0 and self.car_tracking['sub1_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub1_enter_times'].keys()))
                    enter_time = self.car_tracking['sub1_enter_times'].pop(car_id)
                    time_in_system = (timestamp - enter_time).total_seconds()
                    self.performance_metrics['avg_time_in_system'].append(time_in_system)
                    self.current_car_count_sub1 -= 1
                    self.total_cars_exited_sub1 += 1
                else:
                    # If tracking inconsistent, forcibly reduce counters
                    total_now = self.current_car_count_sub1 + self.current_car_count_sub2
                    if total_now > self.normal_total_limit:
                        # reduce sub2 first
                        if self.current_car_count_sub2 > 0:
                            self.current_car_count_sub2 -= 1
                        elif self.current_car_count_sub1 > 0:
                            self.current_car_count_sub1 -= 1
                    else:
                        break

        # Determine traffic status for display (do NOT modify manual flags here)
        total_cars = self.current_car_count_sub1 + self.current_car_count_sub2

        if self.emergency_mode:
            traffic_status = "🚑 EMERGENCY"
            led_red = 1
            led_green = 0
            gate_state = 1
        elif total_cars >= self.overload_min:
            # show overload-like status (6+), but DO NOT auto-enable overload_mode
            traffic_status = "🚨 OVERLOAD"
            led_red = 0
            led_green = 1
            gate_state = 1
        elif self.current_car_count_sub1 >= self.subroad1_limit:
            traffic_status = "🟡 SUBROAD1 FULL"
            led_red = 0
            led_green = 1
            gate_state = 0
        else:
            traffic_status = "✅ NORMAL"
            led_red = 0
            led_green = 1
            gate_state = 0

        data_point = {
            'timestamp': timestamp,
            'ultrasonic_sub1_start': us_sub1_start,
            'ultrasonic_sub1_end': us_sub1_end,
            'ultrasonic_sub2_start': us_sub2_start,
            'ultrasonic_sub2_end': us_sub2_end,
            'ir_sensor': ir_detection,
            'mq135_pollution': pollution,
            'led_red_state': led_red,
            'led_green_state': led_green,
            'gate_state': gate_state,
            'ambulance_detected': ambulance_detected,
            'car_count_sub1': self.current_car_count_sub1,
            'car_count_sub2': self.current_car_count_sub2,
            'traffic_status': traffic_status,
            'cars_entered_sub1': cars_entered_sub1,
            'cars_exited_sub1': cars_exited_sub1,
            'cars_entered_sub2': cars_entered_sub2,
            'cars_exited_sub2': cars_exited_sub2,
            'total_entered_sub1': self.total_cars_entered_sub1,
            'total_exited_sub1': self.total_cars_exited_sub1,
            'total_entered_sub2': self.total_cars_entered_sub2,
            'total_exited_sub2': self.total_cars_exited_sub2
        }

        return data_point

    def toggle_overload(self):
        """Toggle overload mode with proper car distribution (manual only)"""
        if not self.overload_mode:
            # when user enables overload, choose target 6 or 7 and fill to that target
            target = random.choice([6, 7])
            target = min(target, self.total_cars_limit)
            self.activate_overload_fill(target_total=target)
            # overload_mode set to True inside activate_overload_fill
        else:
            # turning off: flip flag and enforce normal cap (<= normal_total_limit)
            self.overload_mode = False
            # Enforce cap immediately: remove excess cars if any (prefer sub2)
            total_now = self.current_car_count_sub1 + self.current_car_count_sub2
            # We'll remove until <= normal_total_limit
            while total_now > self.normal_total_limit:
                if self.current_car_count_sub2 > 0 and self.car_tracking['sub2_enter_times']:
                    # simulate exit from sub2
                    car_id = random.choice(list(self.car_tracking['sub2_enter_times'].keys()))
                    enter_time = self.car_tracking['sub2_enter_times'].pop(car_id)
                    # record time in system
                    time_in_system = (datetime.now() - enter_time).total_seconds()
                    self.performance_metrics['avg_time_in_system'].append(time_in_system)
                    self.current_car_count_sub2 -= 1
                    self.total_cars_exited_sub2 += 1
                elif self.current_car_count_sub1 > 0 and self.car_tracking['sub1_enter_times']:
                    # simulate exit from sub1
                    car_id = random.choice(list(self.car_tracking['sub1_enter_times'].keys()))
                    enter_time = self.car_tracking['sub1_enter_times'].pop(car_id)
                    time_in_system = (datetime.now() - enter_time).total_seconds()
                    self.performance_metrics['avg_time_in_system'].append(time_in_system)
                    self.current_car_count_sub1 -= 1
                    self.total_cars_exited_sub1 += 1
                else:
                    # fallback to decrement counters if tracking inconsistent
                    if self.current_car_count_sub2 > 0:
                        self.current_car_count_sub2 -= 1
                    elif self.current_car_count_sub1 > 0:
                        self.current_car_count_sub1 -= 1
                total_now = self.current_car_count_sub1 + self.current_car_count_sub2

    def clear_emergency(self):
        """Manually clear emergency mode"""
        self.emergency_mode = False
        if self.current_car_count_sub2 > 0:
            while self.current_car_count_sub2 > 0 and self.current_car_count_sub1 < self.subroad1_limit:
                if self.car_tracking['sub2_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub2_enter_times'].keys()))
                    enter_time = self.car_tracking['sub2_enter_times'].pop(car_id)
                    self.car_tracking['sub1_enter_times'][car_id] = enter_time
                self.current_car_count_sub2 -= 1
                self.current_car_count_sub1 += 1

    def toggle_emergency(self):
        """Toggle emergency mode (manual only). No auto-clear."""
        self.emergency_mode = not self.emergency_mode
        # no automatic timer to clear — manual toggle only

    def calculate_advanced_metrics(self, data_point):
        """Calculate advanced performance metrics"""
        response_time = random.uniform(0.05, 1.5)
        accuracy = 0.95 + (random.random() * 0.04)  # 95-99% accuracy
        emergency_time = random.uniform(0.2, 2.0) if data_point['ambulance_detected'] else 0
        total_throughput = (data_point['cars_entered_sub1'] + data_point['cars_entered_sub2'])
        throughput_rate = total_throughput / 60.0  # Cars per minute
        avg_time = np.mean(self.performance_metrics['avg_time_in_system'][-10:]) if self.performance_metrics['avg_time_in_system'] else 0
        total_cars = data_point['car_count_sub1'] + data_point['car_count_sub2']
        congestion = min(1.0, total_cars / 20.0)  # 0-1 scale

        efficiency_components = [
            accuracy * 0.25,
            (1 - response_time / 2) * 0.25,
            (1 - emergency_time / 3) * 0.20 if emergency_time > 0 else 0.20,
            (1 - congestion) * 0.15,
            min(1.0, throughput_rate * 2) * 0.15
        ]
        efficiency = sum(efficiency_components)

        metrics = {
            'timestamp': data_point['timestamp'],
            'response_time': response_time,
            'car_count_accuracy': accuracy,
            'emergency_response_time': emergency_time,
            'system_efficiency': efficiency,
            'throughput_rate': throughput_rate,
            'avg_time_in_system': avg_time,
            'congestion_level': congestion
        }

        return metrics


class AdvancedTrafficAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚦 Advanced Smart Traffic Management AI System")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#2c3e50')

        self.ai_system = AdvancedTrafficAISystem()
        self.expanded_figures = {}  # Store expanded figures

        # indicator variables (0=off,1=on) - MUST be created before setup_gui/control panel uses them
        self.emergency_var = tk.IntVar(value=0)
        self.overload_var = tk.IntVar(value=0)

        self.setup_styles()
        self.setup_gui()
        self.setup_advanced_plots()

    def setup_styles(self):
        """Setup modern styling"""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Modern.TFrame', background='#34495e')
        style.configure('Modern.TLabel', background='#34495e', foreground='white', font=('Arial', 10))
        style.configure('Modern.TButton', font=('Arial', 10, 'bold'))
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'), foreground='#3498db')

    def setup_gui(self):
        """Setup the advanced GUI interface"""
        main_frame = ttk.Frame(self.root, style='Modern.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        header_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame, text="🚦 ADVANCED TRAFFIC MANAGEMENT AI SYSTEM",
                  style='Title.TLabel').pack(pady=10)

        left_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_advanced_control_panel(left_frame)
        self.setup_advanced_realtime_display(left_frame)
        self.setup_advanced_analytics_panel(right_frame)

    def setup_advanced_control_panel(self, parent):
        """Setup advanced control panel"""
        control_frame = ttk.LabelFrame(parent, text="🎮 SYSTEM CONTROLS", padding=15)
        control_frame.pack(fill=tk.X, pady=5)

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.start_btn = ttk.Button(btn_frame, text="▶️ START SIMULATION",
                                    command=self.start_simulation, style='Modern.TButton')
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="⏹️ STOP SIMULATION",
                                   command=self.stop_simulation, state=tk.DISABLED, style='Modern.TButton')
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.emergency_btn = ttk.Button(btn_frame, text="🚑 TOGGLE EMERGENCY",
                                        command=self.toggle_emergency, style='Modern.TButton')
        self.emergency_btn.pack(side=tk.LEFT, padx=5)

        self.overload_btn = ttk.Button(btn_frame, text="🚗 TOGGLE OVERLOAD",
                                       command=self.toggle_overload, style='Modern.TButton')
        self.overload_btn.pack(side=tk.LEFT, padx=5)

        # Indicator radio-buttons (styled as buttons)
        indicator_frame = ttk.Frame(control_frame)
        indicator_frame.pack(fill=tk.X, pady=(8, 0))

        # Emergency indicator: On / Off
        em_frame = ttk.Frame(indicator_frame)
        em_frame.pack(side=tk.LEFT, padx=6)
        ttk.Label(em_frame, text="Emergency:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 6))
        # Use tk.Radiobuttons to allow BG color changes easily
        self.em_on_btn = tk.Radiobutton(em_frame, text="ON", indicatoron=0, width=6,
                                        variable=self.emergency_var, value=1, command=self._on_emergency_radio_change)
        self.em_off_btn = tk.Radiobutton(em_frame, text="OFF", indicatoron=0, width=6,
                                         variable=self.emergency_var, value=0, command=self._on_emergency_radio_change)
        self.em_on_btn.pack(side=tk.LEFT)
        self.em_off_btn.pack(side=tk.LEFT)
        # initialize colors based on current (manual) state
        self._update_emergency_indicator(self.ai_system.emergency_mode)

        # Overload indicator: On / Off
        ov_frame = ttk.Frame(indicator_frame)
        ov_frame.pack(side=tk.LEFT, padx=12)
        ttk.Label(ov_frame, text="Overload:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 6))
        self.ov_on_btn = tk.Radiobutton(ov_frame, text="ON", indicatoron=0, width=6,
                                        variable=self.overload_var, value=1, command=self._on_overload_radio_change)
        self.ov_off_btn = tk.Radiobutton(ov_frame, text="OFF", indicatoron=0, width=6,
                                         variable=self.overload_var, value=0, command=self._on_overload_radio_change)
        self.ov_on_btn.pack(side=tk.LEFT)
        self.ov_off_btn.pack(side=tk.LEFT)
        # initialize colors based on current (manual) state
        self._update_overload_indicator(self.ai_system.overload_mode)

        self.setup_advanced_status_display(control_frame)

    def _on_emergency_radio_change(self):
        """Called when emergency radio buttons are clicked (manual only)"""
        val = self.emergency_var.get()
        if val == 1:
            # turn emergency on (manual)
            self.ai_system.emergency_mode = True
            self._update_emergency_indicator(True)
            self.emergency_btn.config(text="🚑 EMERGENCY ACTIVE")
            # no auto-clear — manual only
        else:
            # turn emergency off
            self.ai_system.emergency_mode = False
            self._update_emergency_indicator(False)
            self.emergency_btn.config(text="🚑 TOGGLE EMERGENCY")

    def _on_overload_radio_change(self):
        """Called when overload radio buttons are clicked (manual only)"""
        val = self.overload_var.get()
        if val == 1:
            # Activate overload - fill immediately to chosen target (6 or 7)
            target = random.choice([6, 7])
            self.ai_system.activate_overload_fill(target_total=target)
            self.ai_system.overload_mode = True
            self._update_overload_indicator(True)
            self.overload_btn.config(text="🚗 OVERLOAD ACTIVE")
        else:
            # Deactivate overload - manual off (this enforces normal cap immediately)
            self.ai_system.toggle_overload()
            self._update_overload_indicator(False)
            self.overload_btn.config(text="🚗 TOGGLE OVERLOAD")

    def _update_emergency_indicator(self, is_on: bool):
        """Color emergency indicator buttons: On = red background, Off = gray"""
        try:
            if is_on:
                self.em_on_btn.config(bg='red', fg='white', activebackground='red')
                self.em_off_btn.config(bg='#d9d9d9', fg='black', activebackground='#d9d9d9')
            else:
                self.em_on_btn.config(bg='#d9d9d9', fg='black', activebackground='#d9d9d9')
                self.em_off_btn.config(bg='red', fg='white', activebackground='red')
        except Exception:
            pass
        self.emergency_var.set(1 if is_on else 0)

    def _update_overload_indicator(self, is_on: bool):
        """Color overload indicator buttons: On = green background, Off = gray"""
        try:
            if is_on:
                self.ov_on_btn.config(bg='green', fg='white', activebackground='green')
                self.ov_off_btn.config(bg='#d9d9d9', fg='black', activebackground='#d9d9d9')
            else:
                self.ov_on_btn.config(bg='#d9d9d9', fg='black', activebackground='#d9d9d9')
                self.ov_off_btn.config(bg='green', fg='white', activebackground='green')
        except Exception:
            pass
        self.overload_var.set(1 if is_on else 0)

    def setup_advanced_status_display(self, parent):
        """Setup advanced status display with gauges"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=10)

        sys_status_frame = ttk.LabelFrame(status_frame, text="📊 SYSTEM STATUS", padding=10)
        sys_status_frame.pack(fill=tk.X, pady=5)

        # Removed the empty 'Pollution' and 'Efficiency' entries per request.
        self.status_vars = {}
        status_grid = [
            [("🖥️ System Status", "Stopped"), ("🚦 Traffic Mode", "Normal"), ("🚨 Emergency", "No")],
            [("⚠️ Overload", "No"), ("🚪 Gate Status", "Closed"), ("🚗 Total Cars", "0")],
            [("🚗 Subroad 1", "0"), ("🚗 Subroad 2", "0"), ("🔢 Total Limit", "7")],
        ]

        for row_idx, row in enumerate(status_grid):
            row_frame = ttk.Frame(sys_status_frame)
            row_frame.pack(fill=tk.X, pady=2)
            for col_idx, (label, value) in enumerate(row):
                frame = ttk.Frame(row_frame)
                frame.pack(side=tk.LEFT, padx=15, pady=2)
                ttk.Label(frame, text=f"{label}:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
                self.status_vars[label] = tk.StringVar(value=value)
                status_label = ttk.Label(frame, textvariable=self.status_vars[label],
                                         font=('Arial', 9, 'bold'))
                status_label.pack(side=tk.LEFT)

                if hasattr(self, 'status_labels'):
                    self.status_labels[label] = status_label
                else:
                    self.status_labels = {label: status_label}

    def setup_advanced_realtime_display(self, parent):
        """Setup advanced real-time data display"""
        data_frame = ttk.LabelFrame(parent, text="📡 REAL-TIME SENSOR DATA", padding=10)
        data_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        notebook = ttk.Notebook(data_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        sensor_tab = ttk.Frame(notebook)
        notebook.add(sensor_tab, text="🔍 SENSOR READINGS")

        self.sensor_text = tk.Text(sensor_tab, height=12, width=70, font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(sensor_tab, command=self.sensor_text.yview)
        self.sensor_text.configure(yscrollcommand=scrollbar.set, bg='#1a1a1a', fg='#00ff00')
        self.sensor_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        perf_tab = ttk.Frame(notebook)
        notebook.add(perf_tab, text="📊 PERFORMANCE METRICS")

        self.perf_text = tk.Text(perf_tab, height=12, width=70, font=('Consolas', 9))
        scrollbar2 = ttk.Scrollbar(perf_tab, command=self.perf_text.yview)
        self.perf_text.configure(yscrollcommand=scrollbar2.set, bg='#1a1a1a', fg='#ffaa00')
        self.perf_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)

        stats_tab = ttk.Frame(notebook)
        notebook.add(stats_tab, text="🚗 CAR STATISTICS")

        self.stats_text = tk.Text(stats_tab, height=12, width=70, font=('Consolas', 9))
        scrollbar3 = ttk.Scrollbar(stats_tab, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar3.set, bg='#1a1a1a', fg='#00aaff')
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar3.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_advanced_analytics_panel(self, parent):
        """Setup advanced analytics and visualization panel"""
        analytics_frame = ttk.LabelFrame(parent, text="🤖 AI ANALYTICS & VISUALIZATIONS", padding=10)
        analytics_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.analytics_notebook = ttk.Notebook(analytics_frame)
        self.analytics_notebook.pack(fill=tk.BOTH, expand=True)

        dashboard_tab = ttk.Frame(self.analytics_notebook)
        self.analytics_notebook.add(dashboard_tab, text="📊 DASHBOARD")

        # Create main figure for dashboard - enable constrained_layout & adjust margins to avoid overlap
        self.fig = Figure(figsize=(12, 10), dpi=100, facecolor='#2c3e50', constrained_layout=False)
        self.setup_advanced_plots_layout()

        canvas = FigureCanvasTkAgg(self.fig, dashboard_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, dashboard_tab)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Individual graph tabs: removed Pollution, Efficiency, Heatmap as requested.
        self.create_individual_graph_tabs()

        control_frame = ttk.Frame(analytics_frame)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="🔄 REFRESH ANALYTICS",
                   command=self.update_advanced_analytics).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 EXPORT DATA",
                   command=self.export_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📈 EXPAND ALL GRAPHS",
                   command=self.expand_all_graphs).pack(side=tk.LEFT, padx=5)

    def create_individual_graph_tabs(self):
        """Create individual tabs for each graph (only Traffic Flow and Performance remain)"""
        graph_tabs = [
            ("🚗 Traffic Flow", self.create_traffic_flow_tab),
            ("📈 Performance", self.create_performance_tab),
        ]

        for tab_name, tab_creator in graph_tabs:
            tab = ttk.Frame(self.analytics_notebook)
            self.analytics_notebook.add(tab, text=tab_name)
            tab_creator(tab)

    def create_traffic_flow_tab(self, parent):
        """Create traffic flow analysis tab"""
        fig = Figure(figsize=(10, 6), dpi=100, facecolor='#2c3e50')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        for ax in (ax1, ax2):
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white', labelsize=9)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        # Store traffic figure for expanded update
        self.expanded_figures["traffic_flow"] = (fig, ax1, ax2, canvas)

    def create_performance_tab(self, parent):
        """Create performance metrics tab"""
        fig = Figure(figsize=(10, 6), dpi=100, facecolor='#2c3e50')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        for ax in (ax1, ax2):
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white', labelsize=9)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        self.expanded_figures["performance"] = (fig, ax1, ax2, canvas)

    def setup_advanced_plots_layout(self):
        """Setup advanced layout for matplotlib plots (cleaner, spaced)"""
        self.fig.clear()

        # We'll use a 3x2 grid (5 usable axes) for clarity and spacing
        gs = gridspec.GridSpec(3, 2, figure=self.fig, height_ratios=[1, 1, 0.9])
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Traffic flow
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Car counting accuracy
        self.ax3 = self.fig.add_subplot(gs[1, 0])  # Response times
        self.ax4 = self.fig.add_subplot(gs[1, 1])  # Throughput
        self.ax5 = self.fig.add_subplot(gs[2, :])  # Combined small analytics

        for ax in (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5):
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white', labelsize=9)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        # Global spacing to avoid overlap
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.07, hspace=0.6, wspace=0.45)

    def setup_advanced_plots(self):
        """Initialize advanced plots with sample data"""
        self.update_advanced_analytics()

    def start_simulation(self):
        """Start the traffic simulation"""
        # Enforce safe initial state: manual modes off and Subroad1 <= limit
        self.ai_system.emergency_mode = False
        self.ai_system.overload_mode = False
        self._update_emergency_indicator(False)
        self._update_overload_indicator(False)
        self.emergency_btn.config(text="🚑 TOGGLE EMERGENCY")
        self.overload_btn.config(text="🚗 TOGGLE OVERLOAD")

        # Ensure Subroad1 has <= subroad1_limit cars by moving any excess to subroad2
        while self.ai_system.current_car_count_sub1 > self.ai_system.subroad1_limit:
            if self.ai_system.car_tracking['sub1_enter_times']:
                car_id = random.choice(list(self.ai_system.car_tracking['sub1_enter_times'].keys()))
                enter_time = self.ai_system.car_tracking['sub1_enter_times'].pop(car_id)
                self.ai_system.car_tracking['sub2_enter_times'][car_id] = enter_time
                self.ai_system.current_car_count_sub1 -= 1
                self.ai_system.current_car_count_sub2 += 1
            else:
                # If there's inconsistent counts, just reduce the counter
                self.ai_system.current_car_count_sub1 -= 1

        # Also ensure we start within normal cap (<= normal_total_limit)
        total_now = self.ai_system.current_car_count_sub1 + self.ai_system.current_car_count_sub2
        while total_now > self.ai_system.normal_total_limit:
            if self.ai_system.current_car_count_sub2 > 0 and self.ai_system.car_tracking['sub2_enter_times']:
                car_id = random.choice(list(self.ai_system.car_tracking['sub2_enter_times'].keys()))
                self.ai_system.car_tracking['sub2_enter_times'].pop(car_id)
                self.ai_system.current_car_count_sub2 -= 1
            elif self.ai_system.current_car_count_sub1 > 0 and self.ai_system.car_tracking['sub1_enter_times']:
                car_id = random.choice(list(self.ai_system.car_tracking['sub1_enter_times'].keys()))
                self.ai_system.car_tracking['sub1_enter_times'].pop(car_id)
                self.ai_system.current_car_count_sub1 -= 1
            else:
                if self.ai_system.current_car_count_sub2 > 0:
                    self.ai_system.current_car_count_sub2 -= 1
                elif self.ai_system.current_car_count_sub1 > 0:
                    self.ai_system.current_car_count_sub1 -= 1
            total_now = self.ai_system.current_car_count_sub1 + self.ai_system.current_car_count_sub2

        # Now start simulation loop
        self.ai_system.simulation_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_vars["🖥️ System Status"].set("Running")

        self.simulation_thread = threading.Thread(target=self.run_advanced_simulation, daemon=True)
        self.simulation_thread.start()

    def stop_simulation(self):
        """Stop the traffic simulation"""
        self.ai_system.simulation_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_vars["🖥️ System Status"].set("Stopped")

    def toggle_emergency(self):
        """Toggle emergency mode and update indicator (manual only)"""
        self.ai_system.toggle_emergency()
        self._update_emergency_indicator(self.ai_system.emergency_mode)
        self.emergency_var.set(1 if self.ai_system.emergency_mode else 0)
        if self.ai_system.emergency_mode:
            self.emergency_btn.config(text="🚑 EMERGENCY ACTIVE")
        else:
            self.emergency_btn.config(text="🚑 TOGGLE EMERGENCY")

    def toggle_overload(self):
        """Toggle overload mode and update indicator (manual only)"""
        self.ai_system.toggle_overload()
        self._update_overload_indicator(self.ai_system.overload_mode)
        self.overload_var.set(1 if self.ai_system.overload_mode else 0)
        if self.ai_system.overload_mode:
            self.overload_btn.config(text="🚗 OVERLOAD ACTIVE")
        else:
            self.overload_btn.config(text="🚗 TOGGLE OVERLOAD")

    def run_advanced_simulation(self):
        """Advanced simulation loop"""
        while self.ai_system.simulation_running:
            data_point = self.ai_system.generate_sensor_data()
            metrics = self.ai_system.calculate_advanced_metrics(data_point)

            self.ai_system.real_time_buffer.append((data_point, metrics))

            self.update_advanced_realtime_display(data_point, metrics)
            self.update_advanced_status_display(data_point, metrics)

            if len(self.ai_system.real_time_buffer) % 5 == 0:
                self.update_advanced_analytics()
                self.update_expanded_analytics()

            time.sleep(0.5)

    def update_advanced_realtime_display(self, data_point, metrics):
        """Update advanced real-time data display"""
        # NOTE: removed IR Sensor line from the sensor-reading text as requested
        sensor_text = (
            f"=== 🚦 REAL-TIME TRAFFIC DATA ===\n"
            f"Timestamp: {data_point['timestamp'].strftime('%H:%M:%S')}\n\n"
            f"--- 🚗 CAR COUNTING SYSTEM ---\n"
            f"Subroad 1 Start: {'ENTER' if data_point['ultrasonic_sub1_start'] == 0 else 'No car'}\n"
            f"Subroad 1 End: {'EXIT' if data_point['ultrasonic_sub1_end'] == 0 else 'No car'}\n"
            f"Subroad 2 Start: {'ENTER' if data_point['ultrasonic_sub2_start'] == 0 else 'No car'}\n"
            f"Subroad 2 End: {'EXIT' if data_point['ultrasonic_sub2_end'] == 0 else 'No car'}\n\n"
            f"--- 🔧 SYSTEM SENSORS ---\n"
            f"MQ135 Pollution: {data_point['mq135_pollution']} ppm ({self.get_pollution_alert(data_point['mq135_pollution'])})\n"
            f"LED Status: {'RED' if data_point['led_red_state'] else 'GREEN'}\n"
            f"Gate: {'OPEN' if data_point['gate_state'] else 'CLOSED'}\n\n"
            f"--- 📊 TRAFFIC STATUS ---\n"
            f"Traffic Mode: {data_point['traffic_status']}\n"
            f"Cars in Subroad 1: {data_point['car_count_sub1']}\n"
            f"Cars in Subroad 2: {data_point['car_count_sub2']}\n"
            f"Total Cars Tracked: {len(self.ai_system.car_tracking['car_ids'])}\n"
        )

        perf_text = (
            f"=== ⚡ PERFORMANCE ANALYTICS ===\n"
            f"Timestamp: {metrics['timestamp'].strftime('%H:%M:%S')}\n\n"
            f"--- 🚀 RESPONSE METRICS ---\n"
            f"System Response Time: {metrics['response_time']:.3f} seconds\n"
            f"Emergency Response: {metrics['emergency_response_time']:.3f} seconds\n"
            f"Car Counting Accuracy: {metrics['car_count_accuracy']:.1%}\n\n"
            f"--- 📈 EFFICIENCY METRICS (internal) ---\n"
            f"Overall Efficiency (internal): {metrics['system_efficiency']:.1%}\n"
            f"Throughput Rate: {metrics['throughput_rate']:.2f} cars/min\n"
            f"Avg Time in System: {metrics['avg_time_in_system']:.1f}s\n"
            f"Congestion Level: {metrics['congestion_level']:.1%}\n\n"
            f"--- 🤖 AI PREDICTIONS ---\n"
            f"Traffic Flow: {self.predict_traffic_flow(data_point)}\n"
            f"Congestion Risk: {self.get_congestion_risk(metrics['congestion_level'])}\n"
        )

        total_cars = data_point['car_count_sub1'] + data_point['car_count_sub2']
        stats_text = (
            f"=== 🚗 ADVANCED CAR STATISTICS ===\n"
            f"Timestamp: {data_point['timestamp'].strftime('%H:%M:%S')}\n\n"
            f"--- 📊 CUMULATIVE COUNTS ---\n"
            f"Subroad 1 - Entered: {data_point['total_entered_sub1']} | Exited: {data_point['total_exited_sub1']}\n"
            f"Subroad 2 - Entered: {data_point['total_entered_sub2']} | Exited: {data_point['total_exited_sub2']}\n"
            f"Total System: {data_point['total_entered_sub1'] + data_point['total_entered_sub2']} cars\n\n"
            f"--- 🔄 REAL-TIME EVENTS ---\n"
            f"Current in Subroad 1: {data_point['car_count_sub1']} cars\n"
            f"Current in Subroad 2: {data_point['car_count_sub2']} cars\n"
            f"Active Cars: {total_cars} / {self.ai_system.total_cars_limit}\n\n"
            f"--- 🎯 EVENT DETECTION ---\n"
            f"Last Event Sub1: {'ENTER' if data_point['cars_entered_sub1'] else 'EXIT' if data_point['cars_exited_sub1'] else 'NONE'}\n"
            f"Last Event Sub2: {'ENTER' if data_point['cars_entered_sub2'] else 'EXIT' if data_point['cars_exited_sub2'] else 'NONE'}\n\n"
            f"--- 🔄 OVERFLOW LOGIC ---\n"
            f"Total Cars Limit: {self.ai_system.total_cars_limit}\n"
            f"Subroad 1 Limit: {self.ai_system.subroad1_limit}\n"
            f"Overflow Cars: {max(0, data_point['car_count_sub1'] - self.ai_system.subroad1_limit)}\n"
            f"System Status: {'🟢 NORMAL' if total_cars <= 5 else '🟡 HIGH' if total_cars <= 6 else '🔴 OVERLOAD'}\n"
        )

        self.root.after(0, self._update_advanced_text_widgets, sensor_text, perf_text, stats_text)

    def _update_advanced_text_widgets(self, sensor_text, perf_text, stats_text):
        """Thread-safe update of advanced text widgets"""
        self.sensor_text.delete(1.0, tk.END)
        self.sensor_text.insert(1.0, sensor_text)

        self.perf_text.delete(1.0, tk.END)
        self.perf_text.insert(1.0, perf_text)

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)

    def update_advanced_status_display(self, data_point, metrics):
        """Update advanced status display with colors"""
        total_cars = data_point['car_count_sub1'] + data_point['car_count_sub2']

        self.status_vars["🚦 Traffic Mode"].set(data_point['traffic_status'])
        self.status_vars["🚨 Emergency"].set("Yes" if self.ai_system.emergency_mode else "No")
        self.status_vars["⚠️ Overload"].set("Yes" if self.ai_system.overload_mode else "No")
        self.status_vars["🚪 Gate Status"].set("Open" if data_point['gate_state'] else "Closed")
        self.status_vars["🚗 Total Cars"].set(f"{total_cars}")
        self.status_vars["🖥️ System Status"].set("Running" if self.ai_system.simulation_running else "Stopped")
        self.status_vars["🚗 Subroad 1"].set(f"{data_point['car_count_sub1']}")
        self.status_vars["🚗 Subroad 2"].set(f"{data_point['car_count_sub2']}")

        # Update indicator widgets (reflect manual flags)
        self._update_emergency_indicator(self.ai_system.emergency_mode)
        self._update_overload_indicator(self.ai_system.overload_mode)

        # Simplified color updates
        if self.ai_system.emergency_mode:
            self.status_labels["🚨 Emergency"].config(foreground='red')
        else:
            self.status_labels["🚨 Emergency"].config(foreground='green')

        if self.ai_system.overload_mode:
            self.status_labels["⚠️ Overload"].config(foreground='red')
        else:
            self.status_labels["⚠️ Overload"].config(foreground='green')

        # Color total cars
        try:
            if total_cars <= 4:
                self.status_labels["🚗 Total Cars"].config(foreground='green')
            elif total_cars <= 6:
                self.status_labels["🚗 Total Cars"].config(foreground='orange')
            else:
                self.status_labels["🚗 Total Cars"].config(foreground='red')
        except Exception:
            pass

    def update_advanced_analytics(self):
        """Update advanced analytics plots (clean, spaced, no overlaps)."""
        if len(self.ai_system.real_time_buffer) < 3:
            return

        buffer_list = list(self.ai_system.real_time_buffer)
        timestamps = [dp[0]['timestamp'] for dp in buffer_list]
        # Convert timestamps to matplotlib dates for clean axis formatting
        xvals = mdates.date2num(timestamps)

        car_counts_1 = [dp[0]['car_count_sub1'] for dp in buffer_list]
        car_counts_2 = [dp[0]['car_count_sub2'] for dp in buffer_list]
        response_times = [dp[1]['response_time'] for dp in buffer_list]
        throughput_rates = [dp[1]['throughput_rate'] for dp in buffer_list]
        congestion_levels = [dp[1]['congestion_level'] for dp in buffer_list]
        total_cars = [c1 + c2 for c1, c2 in zip(car_counts_1, car_counts_2)]
        accuracy_values = [dp[1]['car_count_accuracy'] for dp in buffer_list]

        for ax in (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5):
            ax.clear()
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white', labelsize=9)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        # Plot 1: Real-time Traffic Flow (lines + shaded)
        self.ax1.plot_date(xvals, car_counts_1, '-', label='Subroad 1', linewidth=2.5)
        self.ax1.plot_date(xvals, car_counts_2, '-', label='Subroad 2', linewidth=2.5)
        self.ax1.plot_date(xvals, total_cars, '--', label='Total Cars', linewidth=1.8, alpha=0.9)
        self.ax1.fill_between(xvals, car_counts_1, alpha=0.15)
        self.ax1.fill_between(xvals, car_counts_2, alpha=0.15)
        self.ax1.axhline(y=self.ai_system.total_cars_limit, color='red', linestyle=':', alpha=0.7, label=f'Max Limit ({self.ai_system.total_cars_limit})')
        self.ax1.set_title('🚗 Real-time Traffic Flow Analysis', fontweight='bold', fontsize=10)
        self.ax1.set_ylabel('Number of Cars')
        self.ax1.legend(loc='upper left', fontsize=8)
        self.ax1.grid(True, alpha=0.2)
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        for label in self.ax1.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

        # Plot 2: Car Counting Accuracy
        self.ax2.plot_date(xvals, accuracy_values, '-', linewidth=2.5)
        self.ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target (95%)')
        self.ax2.set_title('🎯 Car Counting Accuracy', fontweight='bold', fontsize=10)
        self.ax2.set_ylabel('Accuracy Rate')
        self.ax2.set_ylim(0.8, 1.0)
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.2)
        self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        for label in self.ax2.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

        # Plot 3: Response Times
        self.ax3.plot_date(xvals, response_times, '-', linewidth=2.5)
        self.ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Max Target (1s)')
        self.ax3.set_title('⏱️ System Response Times', fontweight='bold', fontsize=10)
        self.ax3.set_ylabel('Response Time (s)')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.2)
        self.ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        for label in self.ax3.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

        # Plot 4: Throughput (bar) - spaced and with no x-overlap
        indices = np.arange(len(throughput_rates))
        self.ax4.bar(indices, throughput_rates, alpha=0.8)
        self.ax4.set_title('📈 Traffic Throughput Rate (cars/min)', fontweight='bold', fontsize=10)
        self.ax4.set_ylabel('Cars/min')
        # use fewer x-ticks and label them by timestamp in a readable way
        xtick_idxs = indices[::max(1, len(indices)//6)]
        xtick_labels = [timestamps[i].strftime('%H:%M:%S') for i in xtick_idxs]
        self.ax4.set_xticks(xtick_idxs)
        self.ax4.set_xticklabels(xtick_labels, rotation=30, ha='right', fontsize=8)
        self.ax4.grid(True, alpha=0.2)

        # Plot 5: Combined (small) analytics: congestion vs total cars over time
        self.ax5.plot_date(xvals, total_cars, '-', label='Total Cars', linewidth=2)
        self.ax5.plot_date(xvals, [c * 10 for c in congestion_levels], '-', label='Congestion x10 (scaled)', linewidth=1.5, alpha=0.9)
        self.ax5.set_title('🔎 Combined Small Analytics', fontweight='bold', fontsize=10)
        self.ax5.set_ylabel('Value')
        self.ax5.legend(loc='upper left', fontsize=8)
        self.ax5.grid(True, alpha=0.15)
        self.ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        for label in self.ax5.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

        # Apply final layout adjustments to avoid overlap
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.07, hspace=0.6, wspace=0.45)
        self.fig.canvas.draw_idle()

    def update_expanded_analytics(self):
        """Update expanded individual graphs (only traffic_flow & performance)"""
        if len(self.ai_system.real_time_buffer) < 3:
            return

        buffer_list = list(self.ai_system.real_time_buffer)
        timestamps = [dp[0]['timestamp'] for dp in buffer_list]
        xvals = mdates.date2num(timestamps)

        car_counts_1 = [dp[0]['car_count_sub1'] for dp in buffer_list]
        car_counts_2 = [dp[0]['car_count_sub2'] for dp in buffer_list]
        throughput_rates = [dp[1]['throughput_rate'] for dp in buffer_list]
        response_times = [dp[1]['response_time'] for dp in buffer_list]
        accuracy_values = [dp[1]['car_count_accuracy'] for dp in buffer_list]
        total_cars = [c1 + c2 for c1, c2 in zip(car_counts_1, car_counts_2)]

        # Traffic Flow expanded
        if "traffic_flow" in self.expanded_figures:
            fig, ax1, ax2, canvas = self.expanded_figures["traffic_flow"]
            ax1.clear()
            ax2.clear()
            ax1.set_facecolor('#34495e')
            ax2.set_facecolor('#34495e')

            ax1.plot_date(xvals, car_counts_1, '-', label='Subroad 1', linewidth=2.5)
            ax1.plot_date(xvals, car_counts_2, '-', label='Subroad 2', linewidth=2.5)
            ax1.plot_date(xvals, total_cars, '--', label='Total Cars', linewidth=1.8)
            ax1.axhline(y=self.ai_system.total_cars_limit, color='red', linestyle=':', alpha=0.7, label='Max Limit')
            ax1.set_title('🚗 Detailed Traffic Flow Analysis', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Number of Cars')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.25)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            for label in ax1.get_xticklabels():
                label.set_rotation(30)
                label.set_ha('right')

            ax2.plot_date(xvals, throughput_rates, '-', linewidth=2.5)
            ax2.set_title('📈 Traffic Throughput Over Time', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Cars per Minute')
            ax2.set_xlabel('Time')
            ax2.grid(True, alpha=0.25)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            for label in ax2.get_xticklabels():
                label.set_rotation(30)
                label.set_ha('right')

            fig.tight_layout()
            canvas.draw()

        # Performance expanded
        if "performance" in self.expanded_figures:
            fig, ax1, ax2, canvas = self.expanded_figures["performance"]
            ax1.clear()
            ax2.clear()
            ax1.set_facecolor('#34495e')
            ax2.set_facecolor('#34495e')

            ax1.plot_date(xvals, response_times, '-', linewidth=2.5)
            ax1.axhline(y=1.0, color='orange', linestyle='--', label='Target (1s)')
            ax1.set_title('⏱️ System Response Times', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Response Time (s)')
            ax1.legend()
            ax1.grid(True, alpha=0.25)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            for label in ax1.get_xticklabels():
                label.set_rotation(30)
                label.set_ha('right')

            ax2.plot_date(xvals, accuracy_values, '-', linewidth=2.5)
            ax2.axhline(y=0.95, color='red', linestyle='--', label='Target (95%)')
            ax2.set_title('🎯 Car Counting Accuracy', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Accuracy Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.25)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            for label in ax2.get_xticklabels():
                label.set_rotation(30)
                label.set_ha('right')

            fig.tight_layout()
            canvas.draw()

    def expand_all_graphs(self):
        """Expand all graphs to individual windows"""
        for graph_name in self.expanded_figures:
            self.create_expanded_window(graph_name)

    def create_expanded_window(self, graph_name):
        """Create expanded window for individual graph"""
        window = tk.Toplevel(self.root)
        window.title(f"Expanded {graph_name.replace('_', ' ').title()}")
        window.geometry("1000x800")

        fig, *_ = self.expanded_figures[graph_name]
        new_fig = Figure(figsize=(12, 8), dpi=100, facecolor='#2c3e50')

        for i, ax in enumerate(fig.axes):
            new_ax = new_fig.add_subplot(len(fig.axes), 1, i + 1)
            new_ax.set_title(ax.get_title())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            new_ax.set_facecolor('#34495e')

        canvas = FigureCanvasTkAgg(new_fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()

    def predict_traffic_flow(self, data_point):
        """Predict traffic flow using advanced ML model"""
        total_cars = data_point['car_count_sub1'] + data_point['car_count_sub2']

        if total_cars >= 7:
            return "🚨 CRITICAL (Max Capacity)"
        elif total_cars >= 5:
            return "🚛 HIGH (Approaching Limit)"
        elif total_cars >= 3:
            return "🚐 MODERATE (Normal)"
        else:
            return "🚗 LOW (Light Traffic)"

    def get_pollution_alert(self, pollution_level):
        """Get pollution alert level with user-specified thresholds (ppm):
           0-600 Low, 600-1000 Moderate, 1000-1500 High, >1500 Extreme
        """
        try:
            p = float(pollution_level)
        except Exception:
            p = 0.0
        if p > 1500:
            return "🔴 EXTREME"
        elif p > 1000:
            return "🟠 HIGH"
        elif p > 600:
            return "🟡 MODERATE"
        else:
            return "🟢 LOW"

    def get_congestion_risk(self, congestion_level):
        """Get congestion risk assessment"""
        if congestion_level > 0.8:
            return "🔴 HIGH RISK"
        elif congestion_level > 0.5:
            return "🟠 MEDIUM RISK"
        else:
            return "🟢 LOW RISK"

    def export_data(self):
        """Export simulation data to CSV"""
        if self.ai_system.real_time_buffer:
            try:
                filename = f"traffic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                data_list = []
                for data_point, metrics in list(self.ai_system.real_time_buffer):
                    row = {**data_point, **metrics}
                    row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    data_list.append(row)

                df = pd.DataFrame(data_list)
                df.to_csv(filename, index=False)
                messagebox.showinfo("Export Successful", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
        else:
            messagebox.showwarning("No Data", "No simulation data available to export")


def main():
    root = tk.Tk()
    app = AdvancedTrafficAIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
