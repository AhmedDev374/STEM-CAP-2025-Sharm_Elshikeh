import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score
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
import xgboost as xgb
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

        # Advanced performance metrics with proper initialization
        self.performance_metrics = {
            'response_times': [],
            'car_count_accuracy': [],
            'emergency_response_time': [],
            'pollution_alerts': [],
            'system_efficiency': [],
            'throughput_rate': [],
            'avg_time_in_system': [],
            'congestion_level': [],
            'ambulance_detection_time': []  # Added for ambulance detection timing
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
        self.emergency_mode = False
        self.overload_mode = False
        self.next_car_id = 1

        # Limits & thresholds
        self.total_cars_limit = 7
        self.subroad1_limit = 5
        self.normal_total_limit = 5
        self.overload_min = 6

        # Timing and tracking for metrics
        self.last_ambulance_detection = None
        self.last_response_time_calculation = None
        self.throughput_history = deque(maxlen=10)  # Store last 10 throughput values
        self.response_time_history = deque(maxlen=10)  # Store last 10 response times

        # XGBoost specific data
        self.xgboost_predictions = []
        self.xgboost_actuals = []
        self.xgboost_feature_importances = None
        self.xgboost_metrics = {
            'mse': 0.0,
            'r2': 0.0,
            'mae': 0.0
        }

    def initialize_models(self):
        """Initialize advanced machine learning models"""
        models = {
            'traffic_flow_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'emergency_detector': RandomForestClassifier(n_estimators=50, random_state=42),
            'pollution_analyzer': RandomForestRegressor(n_estimators=80, random_state=42),
            'response_time_predictor': RandomForestRegressor(n_estimators=60, random_state=42),
            'congestion_predictor': RandomForestClassifier(n_estimators=70, random_state=42),
            'xgboost_predictor': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                objective='reg:squarederror'
            )
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
        """
        if target_total is None:
            target_total = random.choice([6, 7])
        target_total = min(target_total, self.total_cars_limit)

        while (self.current_car_count_sub1 + self.current_car_count_sub2) < target_total:
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

        self.overload_mode = True

    def generate_sensor_data(self):
        """Generate realistic sensor data with proper car counting"""
        timestamp = datetime.now()

        # Ultrasonic sensors as car counters
        us_sub1_start = 0 if random.random() > 0.8 else random.randint(100, 300)
        us_sub1_end = 0 if random.random() > 0.85 else random.randint(100, 300)
        us_sub2_start = 0 if random.random() > 0.8 else random.randint(100, 300)
        us_sub2_end = 0 if random.random() > 0.85 else random.randint(100, 300)

        # IR sensor for ambulance detection - FIXED to actually work
        ir_detection = 1 if random.random() > 0.97 else 0

        # Track ambulance detection time
        ambulance_detected = 1 if ir_detection else 0
        if ambulance_detected:
            self.last_ambulance_detection = timestamp

        # MQ135 pollution sensor
        base_pollution = 800
        if self.overload_mode:
            base_pollution += 800
        traffic_effect = (self.current_car_count_sub1 + self.current_car_count_sub2) * 100
        pollution = base_pollution + traffic_effect + random.randint(-50, 150)
        pollution = max(0, min(5000, pollution))

        # Car counting logic
        cars_entered_sub1 = 0
        cars_exited_sub1 = 0
        cars_entered_sub2 = 0
        cars_exited_sub2 = 0

        # Emergency handling
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

        # Determine allowed total based on mode
        allowed_total_max = self.total_cars_limit if self.overload_mode else self.normal_total_limit
        overload_min = self.overload_min if self.overload_mode else 0

        # Subroad 1 entry
        if us_sub1_start == 0 and not self.emergency_mode:
            total_cars_now = self.current_car_count_sub1 + self.current_car_count_sub2
            if self.current_car_count_sub1 < self.subroad1_limit and total_cars_now < allowed_total_max:
                car_id = self.generate_car_id()
                self.car_tracking['sub1_enter_times'][car_id] = timestamp
                self.car_tracking['car_ids'].add(car_id)
                self.total_cars_entered_sub1 += 1
                self.current_car_count_sub1 += 1
                cars_entered_sub1 = 1

        # Subroad 1 exit
        if us_sub1_end == 0 and self.current_car_count_sub1 > 0 and not self.emergency_mode:
            total_before_exit = self.current_car_count_sub1 + self.current_car_count_sub2
            if (not self.overload_mode) or (total_before_exit - 1 >= overload_min):
                if self.car_tracking['sub1_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub1_enter_times'].keys()))
                    enter_time = self.car_tracking['sub1_enter_times'].pop(car_id)
                    time_in_system = (timestamp - enter_time).total_seconds()
                    self.performance_metrics['avg_time_in_system'].append(time_in_system)
                self.total_cars_exited_sub1 += 1
                self.current_car_count_sub1 = max(0, self.current_car_count_sub1 - 1)
                cars_exited_sub1 = 1

        # Subroad 2 entry
        if us_sub2_start == 0 and not self.emergency_mode:
            total_cars_now = self.current_car_count_sub1 + self.current_car_count_sub2
            if self.current_car_count_sub1 >= self.subroad1_limit and total_cars_now < allowed_total_max:
                car_id = self.generate_car_id()
                self.car_tracking['sub2_enter_times'][car_id] = timestamp
                self.car_tracking['car_ids'].add(car_id)
                self.total_cars_entered_sub2 += 1
                self.current_car_count_sub2 += 1
                cars_entered_sub2 = 1

        # Subroad 2 exit
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

        # Handle overload mode transition
        if not self.overload_mode and self.current_car_count_sub2 > 0 and not self.emergency_mode:
            while self.current_car_count_sub2 > 0 and self.current_car_count_sub1 < self.subroad1_limit:
                if self.car_tracking['sub2_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub2_enter_times'].keys()))
                    enter_time = self.car_tracking['sub2_enter_times'].pop(car_id)
                    self.car_tracking['sub1_enter_times'][car_id] = enter_time
                self.current_car_count_sub2 -= 1
                self.current_car_count_sub1 += 1

            while (self.current_car_count_sub1 + self.current_car_count_sub2) > self.normal_total_limit and not self.emergency_mode:
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

        # Determine traffic status
        total_cars = self.current_car_count_sub1 + self.current_car_count_sub2
        if self.emergency_mode:
            traffic_status = "🚑 EMERGENCY"
            led_red = 1
            led_green = 0
            gate_state = 1
        elif total_cars >= self.overload_min:
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
        """Toggle overload mode"""
        if not self.overload_mode:
            target = random.choice([6, 7])
            target = min(target, self.total_cars_limit)
            self.activate_overload_fill(target_total=target)
        else:
            self.overload_mode = False
            total_now = self.current_car_count_sub1 + self.current_car_count_sub2
            while total_now > self.normal_total_limit:
                if self.current_car_count_sub2 > 0 and self.car_tracking['sub2_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub2_enter_times'].keys()))
                    enter_time = self.car_tracking['sub2_enter_times'].pop(car_id)
                    time_in_system = (datetime.now() - enter_time).total_seconds()
                    self.performance_metrics['avg_time_in_system'].append(time_in_system)
                    self.current_car_count_sub2 -= 1
                    self.total_cars_exited_sub2 += 1
                elif self.current_car_count_sub1 > 0 and self.car_tracking['sub1_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub1_enter_times'].keys()))
                    enter_time = self.car_tracking['sub1_enter_times'].pop(car_id)
                    time_in_system = (datetime.now() - enter_time).total_seconds()
                    self.performance_metrics['avg_time_in_system'].append(time_in_system)
                    self.current_car_count_sub1 -= 1
                    self.total_cars_exited_sub1 += 1
                total_now = self.current_car_count_sub1 + self.current_car_count_sub2

    def toggle_emergency(self):
        """Toggle emergency mode"""
        self.emergency_mode = not self.emergency_mode

    def calculate_advanced_metrics(self, data_point):
        """Calculate advanced performance metrics - FIXED implementation"""
        # FIXED: Response time calculation based on system load
        base_response = 0.1
        load_factor = (data_point['car_count_sub1'] + data_point['car_count_sub2']) / self.total_cars_limit
        response_time = base_response + (load_factor * 0.5) + random.uniform(-0.05, 0.05)
        response_time = max(0.05, min(2.0, response_time))

        # Store in history
        self.response_time_history.append(response_time)

        # FIXED: Car counting accuracy based on sensor readings
        sensor_accuracy = 0.95 + (random.random() * 0.04)

        # FIXED: Emergency response time calculation
        emergency_time = 0
        if data_point['ambulance_detected']:
            if self.last_response_time_calculation:
                emergency_time = (datetime.now() - self.last_response_time_calculation).total_seconds()
            else:
                emergency_time = random.uniform(0.2, 1.0)
            self.performance_metrics['ambulance_detection_time'].append(emergency_time)

        # FIXED: Throughput rate calculation (cars per minute)
        total_throughput = (data_point['cars_entered_sub1'] + data_point['cars_entered_sub2'])
        self.throughput_history.append(total_throughput)

        # Calculate average throughput over last 10 readings
        if len(self.throughput_history) > 0:
            avg_throughput = sum(self.throughput_history) / len(self.throughput_history)
            # Convert to cars per minute (simulation step is 0.5 seconds)
            throughput_rate = (avg_throughput / 0.5) * 60
        else:
            throughput_rate = 0

        # FIXED: Average time in system
        avg_time = np.mean(self.performance_metrics['avg_time_in_system'][-10:]) if self.performance_metrics['avg_time_in_system'] else 0

        # FIXED: Congestion level
        total_cars = data_point['car_count_sub1'] + data_point['car_count_sub2']
        congestion = min(1.0, total_cars / self.total_cars_limit)

        # FIXED: System efficiency calculation
        efficiency_components = [
            sensor_accuracy * 0.25,
            (1 - min(response_time, 2.0) / 2) * 0.25,
            (1 - min(emergency_time, 3.0) / 3) * 0.20 if emergency_time > 0 else 0.20,
            (1 - congestion) * 0.15,
            min(1.0, throughput_rate / 30) * 0.15  # Normalized throughput
        ]
        efficiency = sum(efficiency_components)

        metrics = {
            'timestamp': data_point['timestamp'],
            'response_time': response_time,
            'car_count_accuracy': sensor_accuracy,
            'emergency_response_time': emergency_time,
            'system_efficiency': efficiency,
            'throughput_rate': throughput_rate,
            'avg_time_in_system': avg_time,
            'congestion_level': congestion,
            'ambulance_detected_flag': data_point['ambulance_detected']
        }

        self.last_response_time_calculation = datetime.now()
        return metrics

    def train_xgboost_model(self):
        """Train XGBoost model on historical data"""
        if len(self.real_time_buffer) < 10:
            return

        data_list = []
        for data_point, metrics in list(self.real_time_buffer):
            row = {
                'mq135_pollution': data_point['mq135_pollution'],
                'car_count_sub1': data_point['car_count_sub1'],
                'car_count_sub2': data_point['car_count_sub2'],
                'ambulance_detected': data_point['ambulance_detected'],
                'response_time': metrics['response_time'],
                'throughput_rate': metrics['throughput_rate'],
                'congestion_level': metrics['congestion_level']
            }
            data_list.append(row)

        df = pd.DataFrame(data_list)
        features = ['mq135_pollution', 'car_count_sub1', 'car_count_sub2', 'ambulance_detected', 'response_time', 'throughput_rate']
        X = df[features]
        y = df['congestion_level']

        if len(X) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.models['xgboost_predictor'].fit(X_train, y_train)

            # Make predictions
            predictions = self.models['xgboost_predictor'].predict(X_test)

            # Store predictions and actuals
            self.xgboost_predictions = predictions.tolist()
            self.xgboost_actuals = y_test.tolist()

            # Calculate metrics
            if len(predictions) > 0:
                self.xgboost_metrics['mse'] = mean_squared_error(y_test, predictions)
                self.xgboost_metrics['r2'] = r2_score(y_test, predictions)
                self.xgboost_metrics['mae'] = np.mean(np.abs(np.array(y_test) - np.array(predictions)))

            # Get feature importances
            self.xgboost_feature_importances = self.models['xgboost_predictor'].feature_importances_

    def predict_with_xgboost(self, data_point, metrics):
        """Predict congestion using trained XGBoost model"""
        if len(self.real_time_buffer) < 10:
            return 0.0

        features = [
            data_point['mq135_pollution'],
            data_point['car_count_sub1'],
            data_point['car_count_sub2'],
            data_point['ambulance_detected'],
            metrics['response_time'],
            metrics['throughput_rate']
        ]

        features_df = pd.DataFrame([features], columns=['mq135_pollution', 'car_count_sub1', 'car_count_sub2', 'ambulance_detected', 'response_time', 'throughput_rate'])

        try:
            prediction = self.models['xgboost_predictor'].predict(features_df)[0]
            return prediction
        except:
            return 0.0


class AdvancedTrafficAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚦 Advanced Smart Traffic Management AI System")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#2c3e50')

        self.ai_system = AdvancedTrafficAISystem()
        self.expanded_figures = {}

        # Indicator variables
        self.emergency_var = tk.IntVar(value=0)
        self.overload_var = tk.IntVar(value=0)

        # Real-time metrics variables for the new screen
        self.realtime_metrics_vars = {}

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
        self.setup_real_time_sensor_metrics(left_frame)  # NEW: Added real-time sensor metrics screen
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

        indicator_frame = ttk.Frame(control_frame)
        indicator_frame.pack(fill=tk.X, pady=(8, 0))

        # Emergency indicator
        em_frame = ttk.Frame(indicator_frame)
        em_frame.pack(side=tk.LEFT, padx=6)
        ttk.Label(em_frame, text="Emergency:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 6))

        self.em_on_btn = tk.Radiobutton(em_frame, text="ON", indicatoron=0, width=6,
                                        variable=self.emergency_var, value=1, command=self._on_emergency_radio_change)
        self.em_off_btn = tk.Radiobutton(em_frame, text="OFF", indicatoron=0, width=6,
                                         variable=self.emergency_var, value=0, command=self._on_emergency_radio_change)
        self.em_on_btn.pack(side=tk.LEFT)
        self.em_off_btn.pack(side=tk.LEFT)

        # Overload indicator
        ov_frame = ttk.Frame(indicator_frame)
        ov_frame.pack(side=tk.LEFT, padx=12)
        ttk.Label(ov_frame, text="Overload:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 6))

        self.ov_on_btn = tk.Radiobutton(ov_frame, text="ON", indicatoron=0, width=6,
                                        variable=self.overload_var, value=1, command=self._on_overload_radio_change)
        self.ov_off_btn = tk.Radiobutton(ov_frame, text="OFF", indicatoron=0, width=6,
                                         variable=self.overload_var, value=0, command=self._on_overload_radio_change)
        self.ov_on_btn.pack(side=tk.LEFT)
        self.ov_off_btn.pack(side=tk.LEFT)

        self.setup_advanced_status_display(control_frame)

    def setup_real_time_sensor_metrics(self, parent):
        """NEW: Setup real-time sensor metrics screen"""
        metrics_frame = ttk.LabelFrame(parent, text="📊 REAL-TIME SENSOR METRICS", padding=15)
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create a grid for metrics display
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.BOTH, expand=True, pady=10)

        # Define metrics to display
        metrics_definitions = [
            ("🚗 Throughput Rate (cars/min):", "throughput_rate", "0.00"),
            ("🚨 Ambulance Detected:", "ambulance_detected", "No"),
            ("⚡ Response Time (s):", "response_time", "0.000"),
            ("📈 System Efficiency:", "system_efficiency", "0.0%"),
            ("🚦 Traffic Status:", "traffic_status", "NORMAL"),
            ("🏎️ Cars in System:", "total_cars", "0"),
            ("🌫️ Pollution Level:", "pollution_level", "0 ppm"),
            ("🎯 Car Counting Accuracy:", "car_accuracy", "0.0%"),
            ("⏱️ Avg Time in System:", "avg_system_time", "0.0s"),
            ("🚧 Congestion Level:", "congestion_level", "0.0%"),
            ("🚪 Gate Status:", "gate_status", "CLOSED"),
            ("🚥 LED Status:", "led_status", "GREEN"),
            ("🤖 XGBoost Prediction:", "xgboost_pred", "0.000"),
            ("📊 XGBoost R² Score:", "xgboost_r2", "0.000"),
        ]

        # Create metrics display
        self.realtime_metrics_vars = {}
        for i, (label, key, default) in enumerate(metrics_definitions):
            row = i // 3
            col = i % 3

            frame = ttk.Frame(metrics_grid)
            frame.grid(row=row, column=col, padx=15, pady=8, sticky="nsew")

            ttk.Label(frame, text=label, font=('Arial', 9, 'bold')).pack(anchor="w")

            var = tk.StringVar(value=default)
            self.realtime_metrics_vars[key] = var

            metric_label = ttk.Label(frame, textvariable=var,
                                     font=('Arial', 10, 'bold'),
                                     foreground=self.get_metric_color(key, default))
            metric_label.pack(anchor="w")

            # Store label for color updates
            if not hasattr(self, 'metrics_labels'):
                self.metrics_labels = {}
            self.metrics_labels[key] = metric_label

        # Configure grid weights
        for i in range(3):
            metrics_grid.columnconfigure(i, weight=1)
        for i in range((len(metrics_definitions) + 2) // 3):
            metrics_grid.rowconfigure(i, weight=1)

    def get_metric_color(self, metric_key, value):
        """Get color for metric based on its value"""
        if metric_key == 'throughput_rate':
            try:
                rate = float(value.split()[0])
                if rate > 20: return 'green'
                elif rate > 10: return 'orange'
                else: return 'red'
            except: return 'white'

        elif metric_key == 'response_time':
            try:
                rt = float(value.split()[0])
                if rt < 0.5: return 'green'
                elif rt < 1.0: return 'orange'
                else: return 'red'
            except: return 'white'

        elif metric_key == 'ambulance_detected':
            return 'red' if 'Yes' in value else 'green'

        elif metric_key == 'system_efficiency':
            try:
                eff = float(value.replace('%', ''))
                if eff > 80: return 'green'
                elif eff > 60: return 'orange'
                else: return 'red'
            except: return 'white'

        elif metric_key == 'traffic_status':
            if 'EMERGENCY' in value: return 'red'
            elif 'OVERLOAD' in value: return 'orange'
            elif 'FULL' in value: return 'yellow'
            else: return 'green'

        elif metric_key == 'congestion_level':
            try:
                cong = float(value.replace('%', ''))
                if cong > 80: return 'red'
                elif cong > 50: return 'orange'
                else: return 'green'
            except: return 'white'

        return 'white'

    def update_real_time_metrics(self, data_point, metrics, xgboost_prediction=0.0):
        """Update real-time sensor metrics display"""
        # Format values for display
        throughput_text = f"{metrics['throughput_rate']:.2f} cars/min"
        ambulance_text = "Yes 🚑" if data_point['ambulance_detected'] else "No"
        response_text = f"{metrics['response_time']:.3f} s"
        efficiency_text = f"{metrics['system_efficiency']*100:.1f}%"
        traffic_text = data_point['traffic_status']
        total_cars_text = f"{data_point['car_count_sub1'] + data_point['car_count_sub2']} cars"
        pollution_text = f"{data_point['mq135_pollution']} ppm"
        accuracy_text = f"{metrics['car_count_accuracy']*100:.1f}%"
        avg_time_text = f"{metrics['avg_time_in_system']:.1f}s"
        congestion_text = f"{metrics['congestion_level']*100:.1f}%"
        gate_text = "OPEN" if data_point['gate_state'] else "CLOSED"
        led_text = "RED" if data_point['led_red_state'] else "GREEN"
        xgboost_text = f"{xgboost_prediction:.3f}"
        xgboost_r2_text = f"{self.ai_system.xgboost_metrics['r2']:.3f}"

        # Update variables
        updates = {
            'throughput_rate': throughput_text,
            'ambulance_detected': ambulance_text,
            'response_time': response_text,
            'system_efficiency': efficiency_text,
            'traffic_status': traffic_text,
            'total_cars': total_cars_text,
            'pollution_level': pollution_text,
            'car_accuracy': accuracy_text,
            'avg_system_time': avg_time_text,
            'congestion_level': congestion_text,
            'gate_status': gate_text,
            'led_status': led_text,
            'xgboost_pred': xgboost_text,
            'xgboost_r2': xgboost_r2_text,
        }

        for key, value in updates.items():
            if key in self.realtime_metrics_vars:
                self.realtime_metrics_vars[key].set(value)
                # Update colors
                if key in self.metrics_labels:
                    color = self.get_metric_color(key, value)
                    self.metrics_labels[key].config(foreground=color)

    def _on_emergency_radio_change(self):
        """Handle emergency radio button change"""
        val = self.emergency_var.get()
        self.ai_system.emergency_mode = (val == 1)
        self._update_emergency_indicator(self.ai_system.emergency_mode)

    def _on_overload_radio_change(self):
        """Handle overload radio button change"""
        val = self.overload_var.get()
        if val == 1:
            target = random.choice([6, 7])
            self.ai_system.activate_overload_fill(target_total=target)
        else:
            self.ai_system.toggle_overload()
        self._update_overload_indicator(self.ai_system.overload_mode)

    def _update_emergency_indicator(self, is_on):
        """Update emergency indicator colors"""
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

    def _update_overload_indicator(self, is_on):
        """Update overload indicator colors"""
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
        """Setup advanced status display"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=10)

        sys_status_frame = ttk.LabelFrame(status_frame, text="📊 SYSTEM STATUS", padding=10)
        sys_status_frame.pack(fill=tk.X, pady=5)

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
                if not hasattr(self, 'status_labels'):
                    self.status_labels = {}
                self.status_labels[label] = status_label

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

        self.fig = Figure(figsize=(12, 10), dpi=100, facecolor='#2c3e50', constrained_layout=False)
        self.setup_advanced_plots_layout()

        canvas = FigureCanvasTkAgg(self.fig, dashboard_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, dashboard_tab)
        toolbar.update()

        self.create_individual_graph_tabs()

        control_frame = ttk.Frame(analytics_frame)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="🔄 REFRESH ANALYTICS",
                   command=self.update_advanced_analytics).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 EXPORT DATA",
                   command=self.export_data).pack(side=tk.LEFT, padx=5)

    def create_individual_graph_tabs(self):
        """Create individual tabs for each graph"""
        graph_tabs = [
            ("🚗 Traffic Flow", self.create_traffic_flow_tab),
            ("📈 Performance", self.create_performance_tab),
            ("🤖 XGBoost", self.create_xgboost_tab),
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

    def create_xgboost_tab(self, parent):
        """Create XGBoost predictions tab"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True)

        graph_frame = ttk.Frame(main_frame)
        graph_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        fig = Figure(figsize=(10, 6), dpi=100, facecolor='#2c3e50')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        for ax in (ax1, ax2):
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white', labelsize=9)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        canvas = FigureCanvasTkAgg(fig, graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, graph_frame)
        toolbar.update()

        self.expanded_figures["xgboost"] = (fig, ax1, ax2, canvas)

    def setup_advanced_plots_layout(self):
        """Setup advanced layout for matplotlib plots"""
        self.fig.clear()
        gs = gridspec.GridSpec(3, 2, figure=self.fig, height_ratios=[1, 1, 0.9])

        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Traffic flow
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Car counting accuracy
        self.ax3 = self.fig.add_subplot(gs[1, 0])  # Response times
        self.ax4 = self.fig.add_subplot(gs[1, 1])  # Throughput
        self.ax5 = self.fig.add_subplot(gs[2, :])  # Combined analytics

        for ax in (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5):
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white', labelsize=9)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.07, hspace=0.6, wspace=0.45)

    def setup_advanced_plots(self):
        """Initialize advanced plots with sample data"""
        self.update_advanced_analytics()

    def start_simulation(self):
        """Start the traffic simulation"""
        self.ai_system.emergency_mode = False
        self.ai_system.overload_mode = False
        self._update_emergency_indicator(False)
        self._update_overload_indicator(False)

        # Ensure proper initial state
        while self.ai_system.current_car_count_sub1 > self.ai_system.subroad1_limit:
            if self.ai_system.car_tracking['sub1_enter_times']:
                car_id = random.choice(list(self.ai_system.car_tracking['sub1_enter_times'].keys()))
                enter_time = self.ai_system.car_tracking['sub1_enter_times'].pop(car_id)
                self.ai_system.car_tracking['sub2_enter_times'][car_id] = enter_time
                self.ai_system.current_car_count_sub1 -= 1
                self.ai_system.current_car_count_sub2 += 1

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
            total_now = self.ai_system.current_car_count_sub1 + self.ai_system.current_car_count_sub2

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
        """Toggle emergency mode"""
        self.ai_system.toggle_emergency()
        self._update_emergency_indicator(self.ai_system.emergency_mode)

    def toggle_overload(self):
        """Toggle overload mode"""
        self.ai_system.toggle_overload()
        self._update_overload_indicator(self.ai_system.overload_mode)

    def run_advanced_simulation(self):
        """Advanced simulation loop"""
        while self.ai_system.simulation_running:
            data_point = self.ai_system.generate_sensor_data()
            metrics = self.ai_system.calculate_advanced_metrics(data_point)
            self.ai_system.real_time_buffer.append((data_point, metrics))

            # Get XGBoost prediction
            xgboost_prediction = self.ai_system.predict_with_xgboost(data_point, metrics)

            # Update all displays
            self.update_advanced_realtime_display(data_point, metrics)
            self.update_advanced_status_display(data_point, metrics)
            self.update_real_time_metrics(data_point, metrics, xgboost_prediction)  # NEW: Update metrics screen

            if len(self.ai_system.real_time_buffer) % 5 == 0:
                self.ai_system.train_xgboost_model()
                self.update_advanced_analytics()
                self.update_expanded_analytics()

            time.sleep(0.5)

    def update_advanced_realtime_display(self, data_point, metrics):
        """Update advanced real-time data display"""
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
            f"IR Sensor (Ambulance): {'DETECTED 🚑' if data_point['ir_sensor'] else 'No detection'}\n"
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
            f"Emergency Response Time: {metrics['emergency_response_time']:.3f} seconds\n"
            f"Car Counting Accuracy: {metrics['car_count_accuracy']:.1%}\n"
            f"Ambulance Detected: {'YES 🚑' if metrics['ambulance_detected_flag'] else 'NO'}\n\n"
            f"--- 📈 EFFICIENCY METRICS ---\n"
            f"Overall System Efficiency: {metrics['system_efficiency']:.1%}\n"
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

        # Update indicator colors
        self._update_emergency_indicator(self.ai_system.emergency_mode)
        self._update_overload_indicator(self.ai_system.overload_mode)

        # Color updates for status labels
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
        """Update advanced analytics plots"""
        if len(self.ai_system.real_time_buffer) < 3:
            return

        buffer_list = list(self.ai_system.real_time_buffer)
        timestamps = [dp[0]['timestamp'] for dp in buffer_list]
        xvals = mdates.date2num(timestamps)

        car_counts_1 = [dp[0]['car_count_sub1'] for dp in buffer_list]
        car_counts_2 = [dp[0]['car_count_sub2'] for dp in buffer_list]
        response_times = [dp[1]['response_time'] for dp in buffer_list]
        throughput_rates = [dp[1]['throughput_rate'] for dp in buffer_list]
        congestion_levels = [dp[1]['congestion_level'] for dp in buffer_list]
        total_cars = [c1 + c2 for c1, c2 in zip(car_counts_1, car_counts_2)]
        accuracy_values = [dp[1]['car_count_accuracy'] for dp in buffer_list]
        ambulance_detections = [dp[0]['ambulance_detected'] for dp in buffer_list]

        for ax in (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5):
            ax.clear()
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white', labelsize=9)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        # Plot 1: Real-time Traffic Flow
        self.ax1.plot_date(xvals, car_counts_1, '-', label='Subroad 1', linewidth=2.5)
        self.ax1.plot_date(xvals, car_counts_2, '-', label='Subroad 2', linewidth=2.5)
        self.ax1.plot_date(xvals, total_cars, '--', label='Total Cars', linewidth=1.8, alpha=0.9)
        self.ax1.fill_between(xvals, car_counts_1, alpha=0.15)
        self.ax1.fill_between(xvals, car_counts_2, alpha=0.15)
        self.ax1.axhline(y=self.ai_system.total_cars_limit, color='red', linestyle=':', alpha=0.7, label=f'Max Limit ({self.ai_system.total_cars_limit})')

        # Mark ambulance detections
        for i, amb in enumerate(ambulance_detections):
            if amb:
                self.ax1.plot_date(xvals[i], total_cars[i], 'r*', markersize=10, label='Ambulance Detected' if i == 0 else "")

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

        # Plot 3: Response Times (FIXED)
        self.ax3.plot_date(xvals, response_times, '-', linewidth=2.5, label='Response Time')

        # Add ambulance detection markers
        amb_indices = [i for i, amb in enumerate(ambulance_detections) if amb]
        for idx in amb_indices:
            self.ax3.plot_date(xvals[idx], response_times[idx], 'r*', markersize=10)

        self.ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Max Target (1s)')
        self.ax3.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Ideal (0.5s)')
        self.ax3.set_title('⏱️ System Response Times', fontweight='bold', fontsize=10)
        self.ax3.set_ylabel('Response Time (s)')
        self.ax3.legend(loc='upper left', fontsize=8)
        self.ax3.grid(True, alpha=0.2)
        self.ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        for label in self.ax3.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

        # Plot 4: Throughput Rate (FIXED)
        indices = np.arange(len(throughput_rates))
        self.ax4.bar(indices, throughput_rates, alpha=0.8, color='skyblue')
        self.ax4.set_title('📈 Traffic Throughput Rate (cars/min)', fontweight='bold', fontsize=10)
        self.ax4.set_ylabel('Cars/min')
        self.ax4.axhline(y=np.mean(throughput_rates) if throughput_rates else 0,
                         color='red', linestyle='--', alpha=0.7, label=f'Avg: {np.mean(throughput_rates):.1f}')

        xtick_idxs = indices[::max(1, len(indices)//6)]
        xtick_labels = [timestamps[i].strftime('%H:%M:%S') for i in xtick_idxs]
        self.ax4.set_xticks(xtick_idxs)
        self.ax4.set_xticklabels(xtick_labels, rotation=30, ha='right', fontsize=8)
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.2)

        # Plot 5: Combined analytics
        self.ax5.plot_date(xvals, total_cars, '-', label='Total Cars', linewidth=2)
        self.ax5.plot_date(xvals, [c * 10 for c in congestion_levels], '-', label='Congestion x10', linewidth=1.5, alpha=0.9)

        # Add throughput trend
        if len(throughput_rates) > 1:
            throughput_norm = [t/5 for t in throughput_rates]  # Normalize
            self.ax5.plot_date(xvals, throughput_norm, '--', label='Throughput/5', linewidth=1.5, alpha=0.7)

        self.ax5.set_title('🔎 Combined System Analytics', fontweight='bold', fontsize=10)
        self.ax5.set_ylabel('Value')
        self.ax5.legend(loc='upper left', fontsize=8)
        self.ax5.grid(True, alpha=0.15)
        self.ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        for label in self.ax5.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

        self.fig.canvas.draw_idle()

    def update_expanded_analytics(self):
        """Update expanded individual graphs"""
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

        # XGBoost expanded
        if "xgboost" in self.expanded_figures:
            fig, ax1, ax2, canvas = self.expanded_figures["xgboost"]
            ax1.clear()
            ax2.clear()
            ax1.set_facecolor('#34495e')
            ax2.set_facecolor('#34495e')

            if self.ai_system.xgboost_predictions and self.ai_system.xgboost_actuals:
                indices = np.arange(len(self.ai_system.xgboost_predictions))
                ax1.plot(indices, self.ai_system.xgboost_predictions, '-', label='Predicted', linewidth=2.5)
                ax1.plot(indices, self.ai_system.xgboost_actuals, '--', label='Actual', linewidth=2.5)
                ax1.set_title('🤖 XGBoost: Predicted vs Actual Congestion', fontweight='bold', fontsize=12)
                ax1.set_ylabel('Congestion Level')
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.25)

                if self.ai_system.xgboost_feature_importances is not None:
                    features = ['mq135_pollution', 'car_count_sub1', 'car_count_sub2',
                               'ambulance_detected', 'response_time', 'throughput_rate']
                    ax2.barh(features, self.ai_system.xgboost_feature_importances, alpha=0.8)
                    ax2.set_title('📊 XGBoost Feature Importances', fontweight='bold', fontsize=12)
                    ax2.set_xlabel('Importance Score')
                    ax2.grid(True, alpha=0.25)

            fig.tight_layout()
            canvas.draw()

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
        """Get pollution alert level"""
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
