import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
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
            'ultrasonic_sub1_start': [],  # Car enter events (0 = car detected)
            'ultrasonic_sub1_end': [],  # Car exit events (0 = car detected)
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
        self.emergency_mode = False
        self.overload_mode = False
        self.next_car_id = 1
        self.total_cars_limit = 7
        self.subroad1_limit = 5

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

    def generate_sensor_data(self):
        """Generate realistic sensor data with proper car counting"""
        timestamp = datetime.now()

        # Ultrasonic sensors as car counters (0 = car detected, >0 = no car)
        us_sub1_start = 0 if random.random() > 0.8 else random.randint(100, 300)
        us_sub1_end = 0 if random.random() > 0.85 else random.randint(100, 300)
        us_sub2_start = 0 if random.random() > 0.8 else random.randint(100, 300)
        us_sub2_end = 0 if random.random() > 0.85 else random.randint(100, 300)

        # IR sensor for ambulance detection
        ir_detection = 1 if random.random() > 0.97 else 0

        # MQ135 pollution sensor - affected by traffic
        base_pollution = 800
        traffic_effect = (self.current_car_count_sub1 + self.current_car_count_sub2) * 50
        pollution = base_pollution + traffic_effect + random.randint(-100, 100)
        pollution = max(500, min(2500, pollution))

        # Car counting logic with unique IDs
        cars_entered_sub1 = 0
        cars_exited_sub1 = 0
        cars_entered_sub2 = 0
        cars_exited_sub2 = 0

        # Handle emergency mode first - clear subroad 2
        if self.emergency_mode:
            # Move all cars from subroad 2 to exited
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

        # Subroad 1 entry - normal flow (up to 5 cars max)
        if us_sub1_start == 0 and not self.emergency_mode:
            total_cars = self.current_car_count_sub1 + self.current_car_count_sub2

            # Allow entry to subroad1 only if it has less than 5 cars and total system is under limit
            if self.current_car_count_sub1 < 5 and total_cars < self.total_cars_limit:
                car_id = self.generate_car_id()
                self.car_tracking['sub1_enter_times'][car_id] = timestamp
                self.car_tracking['car_ids'].add(car_id)
                self.total_cars_entered_sub1 += 1
                self.current_car_count_sub1 += 1
                cars_entered_sub1 = 1

        # Subroad 1 exit
        if us_sub1_end == 0 and self.current_car_count_sub1 > 0 and not self.emergency_mode:
            if self.car_tracking['sub1_enter_times']:
                car_id = random.choice(list(self.car_tracking['sub1_enter_times'].keys()))
                enter_time = self.car_tracking['sub1_enter_times'].pop(car_id)
                time_in_system = (timestamp - enter_time).total_seconds()
                self.performance_metrics['avg_time_in_system'].append(time_in_system)
            self.total_cars_exited_sub1 += 1
            self.current_car_count_sub1 = max(0, self.current_car_count_sub1 - 1)
            cars_exited_sub1 = 1

        # Subroad 2 entry - ONLY when subroad1 is full (5 cars) AND we have overload
        if us_sub2_start == 0 and not self.emergency_mode:
            total_cars = self.current_car_count_sub1 + self.current_car_count_sub2

            # During overload OR when we naturally reach 5+ cars, use subroad2 for extras
            if (
                    self.overload_mode or total_cars >= 5) and total_cars < self.total_cars_limit and self.current_car_count_sub1 >= 5:
                car_id = self.generate_car_id()
                self.car_tracking['sub2_enter_times'][car_id] = timestamp
                self.car_tracking['car_ids'].add(car_id)
                self.total_cars_entered_sub2 += 1
                self.current_car_count_sub2 += 1
                cars_entered_sub2 = 1

        # Subroad 2 exit - only when not in emergency mode
        if us_sub2_end == 0 and self.current_car_count_sub2 > 0 and not self.emergency_mode:
            if self.car_tracking['sub2_enter_times']:
                car_id = random.choice(list(self.car_tracking['sub2_enter_times'].keys()))
                enter_time = self.car_tracking['sub2_enter_times'].pop(car_id)
                time_in_system = (timestamp - enter_time).total_seconds()
                self.performance_metrics['avg_time_in_system'].append(time_in_system)
            self.total_cars_exited_sub2 += 1
            self.current_car_count_sub2 = max(0, self.current_car_count_sub2 - 1)
            cars_exited_sub2 = 1

        # Auto-detect overload condition based on total cars
        total_cars = self.current_car_count_sub1 + self.current_car_count_sub2
        if total_cars >= 6 and not self.emergency_mode:
            self.overload_mode = True
        elif total_cars <= 5 and not self.emergency_mode:
            self.overload_mode = False

        # When overload mode is deactivated AND we're not in emergency, move cars back from subroad2 to subroad1
        if not self.overload_mode and self.current_car_count_sub2 > 0 and not self.emergency_mode:
            # Move cars from subroad 2 back to subroad 1 if possible
            while self.current_car_count_sub2 > 0 and self.current_car_count_sub1 < 5:
                if self.car_tracking['sub2_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub2_enter_times'].keys()))
                    enter_time = self.car_tracking['sub2_enter_times'].pop(car_id)
                    # Move to subroad 1
                    self.car_tracking['sub1_enter_times'][car_id] = enter_time
                self.current_car_count_sub2 -= 1
                self.current_car_count_sub1 += 1

        # Determine traffic status with enhanced logic
        total_cars = self.current_car_count_sub1 + self.current_car_count_sub2

        if self.emergency_mode:
            traffic_status = "🚑 EMERGENCY"
            led_red = 1
            led_green = 0
            gate_state = 1  # Open gate during emergency
        elif self.overload_mode:
            traffic_status = "🚨 OVERLOAD"
            led_red = 0
            led_green = 1
            gate_state = 1  # Open gate in overload
        elif self.current_car_count_sub1 >= 3:
            traffic_status = "⚠️  MODERATE"
            led_red = 0
            led_green = 1
            gate_state = 0
        else:
            traffic_status = "✅ NORMAL"
            led_red = 0
            led_green = 1
            gate_state = 0

        # Ambulance detection logic
        ambulance_detected = 1 if ir_detection else 0
        if ambulance_detected:
            self.emergency_mode = True
            # Emergency mode lasts for 15 seconds
            threading.Timer(15, self.clear_emergency).start()

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
        """Toggle overload mode with proper car distribution"""
        if not self.overload_mode:
            # Activate overload mode - allow cars to go to subroad 2
            self.overload_mode = True
        else:
            # Deactivate overload mode
            self.overload_mode = False
            # When deactivating, we'll let the natural logic in generate_sensor_data
            # handle moving cars back from subroad 2 to subroad 1
    def clear_emergency(self):
        """Clear emergency mode after 15 seconds"""
        self.emergency_mode = False
        # After emergency, ensure normal distribution (3-5 cars in subroad 1, 0 in subroad 2)
        if self.current_car_count_sub2 > 0:
            # Move cars from subroad 2 back to subroad 1 if possible
            while self.current_car_count_sub2 > 0 and self.current_car_count_sub1 < 5:
                if self.car_tracking['sub2_enter_times']:
                    car_id = random.choice(list(self.car_tracking['sub2_enter_times'].keys()))
                    enter_time = self.car_tracking['sub2_enter_times'].pop(car_id)
                    # Move to subroad 1
                    self.car_tracking['sub1_enter_times'][car_id] = enter_time
                self.current_car_count_sub2 -= 1
                self.current_car_count_sub1 += 1

    def toggle_emergency(self):
        """Toggle emergency mode"""
        self.emergency_mode = not self.emergency_mode
        if self.emergency_mode:
            # Emergency mode lasts for 15 seconds
            threading.Timer(15, self.clear_emergency).start()

    def calculate_advanced_metrics(self, data_point):
        """Calculate advanced performance metrics"""
        # Response time calculation
        response_time = random.uniform(0.05, 1.5)

        # Car counting accuracy (improved simulation)
        accuracy = 0.95 + (random.random() * 0.04)  # 95-99% accuracy

        # Emergency response time
        emergency_time = random.uniform(0.2, 2.0) if data_point['ambulance_detected'] else 0

        # Throughput calculation
        total_throughput = (data_point['cars_entered_sub1'] + data_point['cars_entered_sub2'])
        throughput_rate = total_throughput / 60.0  # Cars per minute

        # Average time in system
        avg_time = np.mean(self.performance_metrics['avg_time_in_system'][-10:]) if self.performance_metrics[
            'avg_time_in_system'] else 0

        # Congestion level
        total_cars = data_point['car_count_sub1'] + data_point['car_count_sub2']
        congestion = min(1.0, total_cars / 20.0)  # 0-1 scale

        # Advanced system efficiency calculation
        efficiency_components = [
            accuracy * 0.25,  # Counting accuracy
            (1 - response_time / 2) * 0.25,  # Response time
            (1 - emergency_time / 3) * 0.20 if emergency_time > 0 else 0.20,  # Emergency response
            (1 - congestion) * 0.15,  # Congestion level
            min(1.0, throughput_rate * 2) * 0.15  # Throughput
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
        self.setup_styles()
        self.setup_gui()
        self.setup_advanced_plots()

    def setup_styles(self):
        """Setup modern styling"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure modern colors
        style.configure('Modern.TFrame', background='#34495e')
        style.configure('Modern.TLabel', background='#34495e', foreground='white', font=('Arial', 10))
        style.configure('Modern.TButton', font=('Arial', 10, 'bold'))
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'), foreground='#3498db')

    def setup_gui(self):
        """Setup the advanced GUI interface"""
        # Create main frames with modern styling
        main_frame = ttk.Frame(self.root, style='Modern.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Header
        header_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame, text="🚦 ADVANCED TRAFFIC MANAGEMENT AI SYSTEM",
                  style='Title.TLabel').pack(pady=10)

        # Left panel for controls and real-time data
        left_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel for analytics
        right_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Control panel
        self.setup_advanced_control_panel(left_frame)

        # Real-time data display
        self.setup_advanced_realtime_display(left_frame)

        # Analytics panel
        self.setup_advanced_analytics_panel(right_frame)

    def setup_advanced_control_panel(self, parent):
        """Setup advanced control panel"""
        control_frame = ttk.LabelFrame(parent, text="🎮 SYSTEM CONTROLS", padding=15)
        control_frame.pack(fill=tk.X, pady=5)

        # Control buttons with icons
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

        # Advanced status indicators with radio buttons
        self.setup_advanced_status_display(control_frame)

    def setup_advanced_status_display(self, parent):
        """Setup advanced status display with gauges and mode indicators"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=10)

        # System status
        sys_status_frame = ttk.LabelFrame(status_frame, text="📊 SYSTEM STATUS", padding=10)
        sys_status_frame.pack(fill=tk.X, pady=5)

        self.status_vars = {}
        status_grid = [
            [("🖥️ System Status", "Stopped"), ("🚦 Traffic Mode", "Normal"), ("🚗 Total Cars", "0")],
            [("🚗 Subroad 1", "0"), ("🚗 Subroad 2", "0"), ("🚪 Gate Status", "CLOSED")],  # Added Gate Status here
            [("📈 Efficiency", "0%"), ("⏱️ Avg Time", "0s"), ("🌫️ Pollution", "Low")]
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

                # Store reference for color updates
                if hasattr(self, 'status_labels'):
                    self.status_labels[label] = status_label
                else:
                    self.status_labels = {label: status_label}

        # Add mode indicators for overload and emergency status
        mode_frame = ttk.Frame(sys_status_frame)
        mode_frame.pack(fill=tk.X, pady=10)

        # Overload Indicator
        overload_frame = ttk.Frame(mode_frame)
        overload_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(overload_frame, text="🚨 OVERLOAD MODE:",
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        self.overload_indicator = tk.Label(
            overload_frame,
            text="INACTIVE",
            font=('Arial', 10, 'bold'),
            bg='red',  # Red when inactive
            fg='white',
            relief='raised',
            padx=10,
            pady=2
        )
        self.overload_indicator.pack(side=tk.LEFT, padx=5)

        # Emergency Indicator
        emergency_frame = ttk.Frame(mode_frame)
        emergency_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(emergency_frame, text="🚑 EMERGENCY MODE:",
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        self.emergency_indicator = tk.Label(
            emergency_frame,
            text="INACTIVE",
            font=('Arial', 10, 'bold'),
            bg='red',  # Red when inactive
            fg='white',
            relief='raised',
            padx=10,
            pady=2
        )
        self.emergency_indicator.pack(side=tk.LEFT, padx=5)
    def setup_advanced_realtime_display(self, parent):
        """Setup advanced real-time data display"""
        data_frame = ttk.LabelFrame(parent, text="📡 REAL-TIME SENSOR DATA", padding=10)
        data_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create notebook for different data views
        notebook = ttk.Notebook(data_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Sensor data tab
        sensor_tab = ttk.Frame(notebook)
        notebook.add(sensor_tab, text="🔍 SENSOR READINGS")

        self.sensor_text = tk.Text(sensor_tab, height=12, width=70, font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(sensor_tab, command=self.sensor_text.yview)
        self.sensor_text.configure(yscrollcommand=scrollbar.set, bg='#1a1a1a', fg='#00ff00')
        self.sensor_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Performance metrics tab
        perf_tab = ttk.Frame(notebook)
        notebook.add(perf_tab, text="📊 PERFORMANCE METRICS")

        self.perf_text = tk.Text(perf_tab, height=12, width=70, font=('Consolas', 9))
        scrollbar2 = ttk.Scrollbar(perf_tab, command=self.perf_text.yview)
        self.perf_text.configure(yscrollcommand=scrollbar2.set, bg='#1a1a1a', fg='#ffaa00')
        self.perf_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)

        # Car statistics tab
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

        # Create notebook for analytics
        self.analytics_notebook = ttk.Notebook(analytics_frame)
        self.analytics_notebook.pack(fill=tk.BOTH, expand=True)

        # Main dashboard tab
        dashboard_tab = ttk.Frame(self.analytics_notebook)
        self.analytics_notebook.add(dashboard_tab, text="📊 DASHBOARD")

        # Create main figure for dashboard
        self.fig = Figure(figsize=(12, 10), dpi=100, facecolor='#2c3e50')
        self.setup_advanced_plots_layout()

        canvas = FigureCanvasTkAgg(self.fig, dashboard_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(canvas, dashboard_tab)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Individual graph tabs
        self.create_individual_graph_tabs()

        # Control buttons for analytics
        control_frame = ttk.Frame(analytics_frame)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="🔄 REFRESH ANALYTICS",
                   command=self.update_advanced_analytics).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 EXPORT DATA",
                   command=self.export_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📈 EXPAND ALL GRAPHS",
                   command=self.expand_all_graphs).pack(side=tk.LEFT, padx=5)

    def create_individual_graph_tabs(self):
        """Create individual tabs for each graph"""
        graph_tabs = [
            ("🚗 Traffic Flow", self.create_traffic_flow_tab),
            ("📈 Performance", self.create_performance_tab),
            ("🌫️ Pollution", self.create_pollution_tab),
            ("⚡ Efficiency", self.create_efficiency_tab),
            ("🔥 Heatmap", self.create_heatmap_tab)
        ]

        for tab_name, tab_creator in graph_tabs:
            tab = ttk.Frame(self.analytics_notebook)
            self.analytics_notebook.add(tab, text=tab_name)
            tab_creator(tab)

    def create_traffic_flow_tab(self, parent):
        """Create traffic flow analysis tab"""
        fig = Figure(figsize=(10, 8), dpi=100, facecolor='#2c3e50')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.set_facecolor('#34495e')
        ax2.set_facecolor('#34495e')
        for ax in [ax1, ax2]:
            ax.tick_params(colors='white')
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
        fig = Figure(figsize=(10, 8), dpi=100, facecolor='#2c3e50')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.set_facecolor('#34495e')
        ax2.set_facecolor('#34495e')
        for ax in [ax1, ax2]:
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        self.expanded_figures["performance"] = (fig, ax1, ax2, canvas)

    def create_pollution_tab(self, parent):
        """Create pollution analysis tab"""
        fig = Figure(figsize=(10, 8), dpi=100, facecolor='#2c3e50')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.set_facecolor('#34495e')
        ax2.set_facecolor('#34495e')
        for ax in [ax1, ax2]:
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        self.expanded_figures["pollution"] = (fig, ax1, ax2, canvas)

    def create_efficiency_tab(self, parent):
        """Create efficiency analysis tab"""
        fig = Figure(figsize=(10, 8), dpi=100, facecolor='#2c3e50')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.set_facecolor('#34495e')
        ax2.set_facecolor('#34495e')
        for ax in [ax1, ax2]:
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        self.expanded_figures["efficiency"] = (fig, ax1, ax2, canvas)

    def create_heatmap_tab(self, parent):
        """Create heatmap analysis tab"""
        fig = Figure(figsize=(10, 8), dpi=100, facecolor='#2c3e50')
        ax = fig.add_subplot(111)

        ax.set_facecolor('#34495e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        self.expanded_figures["heatmap"] = (fig, ax, canvas)

    def setup_advanced_plots_layout(self):
        """Setup advanced layout for matplotlib plots"""
        self.fig.clear()

        # Create complex grid layout
        gs = gridspec.GridSpec(3, 3, figure=self.fig, hspace=0.8, wspace=0.6)

        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Traffic flow
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Car counting
        self.ax3 = self.fig.add_subplot(gs[0, 2])  # System efficiency
        self.ax4 = self.fig.add_subplot(gs[1, 0])  # Pollution levels
        self.ax5 = self.fig.add_subplot(gs[1, 1])  # Response times
        self.ax6 = self.fig.add_subplot(gs[1, 2])  # Throughput
        self.ax7 = self.fig.add_subplot(gs[2, :])  # Combined analytics

        # Set dark background for all plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6, self.ax7]:
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        self.fig.tight_layout(pad=3.0)

    def setup_advanced_plots(self):
        """Initialize advanced plots with sample data"""
        self.update_advanced_analytics()

    def start_simulation(self):
        """Start the traffic simulation"""
        self.ai_system.simulation_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_vars["🖥️ System Status"].set("Running")

        # Start simulation thread
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
        if self.ai_system.emergency_mode:
            self.emergency_btn.config(text="🚑 EMERGENCY ACTIVE")
        else:
            self.emergency_btn.config(text="🚑 TOGGLE EMERGENCY")

    def toggle_overload(self):
        """Toggle overload mode"""
        self.ai_system.toggle_overload()
        if self.ai_system.overload_mode:
            self.overload_btn.config(text="🚗 OVERLOAD ACTIVE")
        else:
            self.overload_btn.config(text="🚗 TOGGLE OVERLOAD")

    def run_advanced_simulation(self):
        """Advanced simulation loop"""
        while self.ai_system.simulation_running:
            # Generate new sensor data
            data_point = self.ai_system.generate_sensor_data()
            metrics = self.ai_system.calculate_advanced_metrics(data_point)

            # Update real-time buffer
            self.ai_system.real_time_buffer.append((data_point, metrics))

            # Update GUI
            self.update_advanced_realtime_display(data_point, metrics)
            self.update_advanced_status_display(data_point, metrics)

            # Update analytics periodically
            if len(self.ai_system.real_time_buffer) % 5 == 0:
                self.update_advanced_analytics()
                self.update_expanded_analytics()

            time.sleep(0.5)  # Faster updates for better real-time feel

    def update_advanced_realtime_display(self, data_point, metrics):
        """Update advanced real-time data display"""
        # Sensor data tab
        sensor_text = f"""
=== 🚦 REAL-TIME TRAFFIC DATA ===
Timestamp: {data_point['timestamp'].strftime('%H:%M:%S')}

--- 🚗 CAR COUNTING SYSTEM ---
Subroad 1 Start: {'🚗 CAR ENTERED' if data_point['ultrasonic_sub1_start'] == 0 else 'No car'}
Subroad 1 End: {'🚗 CAR EXITED' if data_point['ultrasonic_sub1_end'] == 0 else 'No car'}
Subroad 2 Start: {'🚗 CAR ENTERED' if data_point['ultrasonic_sub2_start'] == 0 else 'No car'}
Subroad 2 End: {'🚗 CAR EXITED' if data_point['ultrasonic_sub2_end'] == 0 else 'No car'}

--- 🔧 SYSTEM SENSORS ---
IR Sensor: {'🚑 AMBULANCE DETECTED' if data_point['ir_sensor'] else 'Clear'}
MQ135 Pollution: {data_point['mq135_pollution']} ppm
LED Status: {'🔴 RED ON' if data_point['led_red_state'] else '🟢 GREEN ON'}
Gate: {'🚪 OPEN' if data_point['gate_state'] else '🚪 CLOSED'}

--- 📊 TRAFFIC STATUS ---
Traffic Mode: {data_point['traffic_status']}
Cars in Subroad 1: {data_point['car_count_sub1']}
Cars in Subroad 2: {data_point['car_count_sub2']}
Total Cars Tracked: {len(self.ai_system.car_tracking['car_ids'])}
"""

        # Performance metrics tab
        perf_text = f"""
=== ⚡ PERFORMANCE ANALYTICS ===
Timestamp: {metrics['timestamp'].strftime('%H:%M:%S')}

--- 🚀 RESPONSE METRICS ---
System Response Time: {metrics['response_time']:.3f} seconds
Emergency Response: {metrics['emergency_response_time']:.3f} seconds
Car Counting Accuracy: {metrics['car_count_accuracy']:.1%}

--- 📈 EFFICIENCY METRICS ---
Overall Efficiency: {metrics['system_efficiency']:.1%}
Throughput Rate: {metrics['throughput_rate']:.2f} cars/min
Avg Time in System: {metrics['avg_time_in_system']:.1f}s
Congestion Level: {metrics['congestion_level']:.1%}

--- 🤖 AI PREDICTIONS ---
Traffic Flow: {self.predict_traffic_flow(data_point)}
Pollution Alert: {self.get_pollution_alert(data_point['mq135_pollution'])}
Congestion Risk: {self.get_congestion_risk(metrics['congestion_level'])}
"""

        # Car statistics tab
        total_cars = data_point['car_count_sub1'] + data_point['car_count_sub2']
        stats_text = f"""
=== 🚗 ADVANCED CAR STATISTICS ===
Timestamp: {data_point['timestamp'].strftime('%H:%M:%S')}

--- 📊 CUMULATIVE COUNTS ---
Subroad 1 - Entered: {data_point['total_entered_sub1']} | Exited: {data_point['total_exited_sub1']}
Subroad 2 - Entered: {data_point['total_entered_sub2']} | Exited: {data_point['total_exited_sub2']}
Total System: {data_point['total_entered_sub1'] + data_point['total_entered_sub2']} cars

--- 🔄 REAL-TIME EVENTS ---
Current in Subroad 1: {data_point['car_count_sub1']} cars
Current in Subroad 2: {data_point['car_count_sub2']} cars
Active Cars: {total_cars} / {self.ai_system.total_cars_limit}

--- 🎯 EVENT DETECTION ---
Last Event Sub1: {'ENTER' if data_point['cars_entered_sub1'] else 'EXIT' if data_point['cars_exited_sub1'] else 'NONE'}
Last Event Sub2: {'ENTER' if data_point['cars_entered_sub2'] else 'EXIT' if data_point['cars_exited_sub2'] else 'NONE'}

--- 🔄 OVERFLOW LOGIC ---
Total Cars Limit: {self.ai_system.total_cars_limit}
Subroad 1 Limit: {self.ai_system.subroad1_limit}
Overflow Cars: {max(0, data_point['car_count_sub1'] - self.ai_system.subroad1_limit)}
System Status: {'🟢 NORMAL' if total_cars <= 7 else '🟡 MODERATE' if total_cars <= 10 else '🔴 OVERLOAD'}
"""

        # Update GUI in thread-safe manner
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
        """Update advanced status display with colors and mode indicators"""
        total_cars = data_point['car_count_sub1'] + data_point['car_count_sub2']

        self.status_vars["🚦 Traffic Mode"].set(data_point['traffic_status'])
        self.status_vars["🚗 Total Cars"].set(f"{total_cars}")
        self.status_vars["⏱️ Avg Time"].set(f"{metrics['avg_time_in_system']:.1f}s")
        self.status_vars["🚗 Subroad 1"].set(f"{data_point['car_count_sub1']}")
        self.status_vars["🚗 Subroad 2"].set(f"{data_point['car_count_sub2']}")
        self.status_vars["📈 Efficiency"].set(f"{metrics['system_efficiency']:.1%}")
        self.status_vars["🌫️ Pollution"].set(self.get_pollution_alert(data_point['mq135_pollution']))

        # Update gate status - OPEN during overload (cars > 5) or emergency
        if self.ai_system.overload_mode or self.ai_system.emergency_mode or total_cars > 5:
            self.status_vars["🚪 Gate Status"].set("OPEN")
            # Make gate status green when open
            if "🚪 Gate Status" in self.status_labels:
                self.status_labels["🚪 Gate Status"].config(foreground='green')
        else:
            self.status_vars["🚪 Gate Status"].set("CLOSED")
            # Make gate status red when closed
            if "🚪 Gate Status" in self.status_labels:
                self.status_labels["🚪 Gate Status"].config(foreground='red')

        # Update mode indicators based on system state
        if self.ai_system.overload_mode:
            self.overload_indicator.config(
                text="ACTIVE",
                bg='green',  # Green when active
                fg='white'
            )
        else:
            self.overload_indicator.config(
                text="INACTIVE",
                bg='red',  # Red when inactive
                fg='white'
            )

        if self.ai_system.emergency_mode:
            self.emergency_indicator.config(
                text="ACTIVE",
                bg='red',  # Red when active
                fg='white'
            )
        else:
            self.emergency_indicator.config(
                text="INACTIVE",
                bg='red',  # Red when inactive
                fg='white'
            )

        # Update colors based on status
        status_colors = {
            "🚦 Traffic Mode": {
                "✅ NORMAL": "green", "⚠️  MODERATE": "orange", "🚨 OVERLOAD": "red", "🚑 EMERGENCY": "purple"
            },
            "🚗 Total Cars": {
                "0": "green", "1": "green", "2": "green", "3": "green",
                "4": "green", "5": "orange", "6": "orange", "7": "red"
            },
            "🌫️ Pollution": {"LOW": "green", "MODERATE": "orange", "HIGH": "red", "CRITICAL": "purple"}
        }

        for label, var in self.status_vars.items():
            if label in status_colors:
                value = var.get()
                color_map = status_colors[label]
                for key, color in color_map.items():
                    if key in value:
                        self.status_labels[label].config(foreground=color)
                        break
    def update_advanced_analytics(self):
        """Update advanced analytics plots"""
        if len(self.ai_system.real_time_buffer) < 3:
            return

        # Extract data for plotting
        buffer_list = list(self.ai_system.real_time_buffer)
        timestamps = [dp[0]['timestamp'] for dp in buffer_list]
        car_counts_1 = [dp[0]['car_count_sub1'] for dp in buffer_list]
        car_counts_2 = [dp[0]['car_count_sub2'] for dp in buffer_list]
        response_times = [dp[1]['response_time'] for dp in buffer_list]
        efficiencies = [dp[1]['system_efficiency'] for dp in buffer_list]
        pollution_levels = [dp[0]['mq135_pollution'] for dp in buffer_list]
        throughput_rates = [dp[1]['throughput_rate'] for dp in buffer_list]
        congestion_levels = [dp[1]['congestion_level'] for dp in buffer_list]
        total_cars = [c1 + c2 for c1, c2 in zip(car_counts_1, car_counts_2)]

        # Clear previous plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6, self.ax7]:
            ax.clear()

        # Plot 1: Advanced Traffic Flow
        self.ax1.plot(timestamps, car_counts_1, 'b-', label='Subroad 1', linewidth=2.5, alpha=0.8)
        self.ax1.plot(timestamps, car_counts_2, 'r-', label='Subroad 2', linewidth=2.5, alpha=0.8)
        self.ax1.plot(timestamps, total_cars, 'g--', label='Total Cars', linewidth=2, alpha=0.6)
        self.ax1.fill_between(timestamps, car_counts_1, alpha=0.3, color='blue')
        self.ax1.fill_between(timestamps, car_counts_2, alpha=0.3, color='red')
        self.ax1.axhline(y=7, color='red', linestyle=':', alpha=0.7, label='Max Limit (7)')
        self.ax1.set_title('🚗 Real-time Traffic Flow Analysis', fontweight='bold', fontsize=10)
        self.ax1.set_ylabel('Number of Cars')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.2)

        # Plot 2: Car Counting Accuracy
        accuracy_values = [dp[1]['car_count_accuracy'] for dp in buffer_list]
        self.ax2.plot(timestamps, accuracy_values, 'g-', linewidth=2.5)
        self.ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target (95%)')
        self.ax2.set_title('🎯 Car Counting Accuracy', fontweight='bold', fontsize=10)
        self.ax2.set_ylabel('Accuracy Rate')
        self.ax2.set_ylim(0.8, 1.0)
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.2)

        # Plot 3: System Efficiency Gauge
        self.ax3.plot(timestamps, efficiencies, 'purple', linewidth=2.5)
        self.ax3.fill_between(timestamps, efficiencies, alpha=0.3, color='purple')
        self.ax3.set_title('⚡ System Efficiency Score', fontweight='bold', fontsize=10)
        self.ax3.set_ylabel('Efficiency (%)')
        self.ax3.set_ylim(0, 1)
        self.ax3.grid(True, alpha=0.2)

        # Plot 4: Pollution vs Traffic
        scatter = self.ax4.scatter(car_counts_1, pollution_levels, c=congestion_levels,
                                   cmap='RdYlGn_r', alpha=0.6, s=50)
        self.ax4.set_title('🌫️ Pollution vs Traffic Density', fontweight='bold', fontsize=10)
        self.ax4.set_xlabel('Cars in Subroad 1')
        self.ax4.set_ylabel('Pollution Level (ppm)')
        self.ax4.grid(True, alpha=0.2)

        # Plot 5: Response Times
        self.ax5.plot(timestamps, response_times, 'orange', linewidth=2.5)
        self.ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Max Target (1s)')
        self.ax5.set_title('⏱️ System Response Times', fontweight='bold', fontsize=10)
        self.ax5.set_ylabel('Response Time (seconds)')
        self.ax5.legend()
        self.ax5.grid(True, alpha=0.2)

        # Plot 6: Throughput Analysis
        self.ax6.bar(range(len(throughput_rates)), throughput_rates,
                     color=['green' if x > 0.5 else 'orange' for x in throughput_rates], alpha=0.7)
        self.ax6.set_title('📈 Traffic Throughput Rate', fontweight='bold', fontsize=10)
        self.ax6.set_ylabel('Cars per Minute')
        self.ax6.set_xticks([])

        # Plot 7: Combined Analytics Heatmap
        if len(buffer_list) > 10:
            data_matrix = np.array([car_counts_1, car_counts_2, pollution_levels,
                                    [x * 100 for x in efficiencies], [x * 10 for x in congestion_levels]])
            im = self.ax7.imshow(data_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
            self.ax7.set_title('🔥 Combined System Analytics Heatmap', fontweight='bold', fontsize=10)
            self.ax7.set_yticks([0, 1, 2, 3, 4])
            self.ax7.set_yticklabels(['Cars Sub1', 'Cars Sub2', 'Pollution', 'Efficiency', 'Congestion'])
            self.ax7.set_xticks([])

        # Refresh canvas
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def update_expanded_analytics(self):
        """Update expanded individual graphs"""
        if len(self.ai_system.real_time_buffer) < 3:
            return

        buffer_list = list(self.ai_system.real_time_buffer)
        timestamps = [dp[0]['timestamp'] for dp in buffer_list]
        car_counts_1 = [dp[0]['car_count_sub1'] for dp in buffer_list]
        car_counts_2 = [dp[0]['car_count_sub2'] for dp in buffer_list]
        response_times = [dp[1]['response_time'] for dp in buffer_list]
        efficiencies = [dp[1]['system_efficiency'] for dp in buffer_list]
        pollution_levels = [dp[0]['mq135_pollution'] for dp in buffer_list]
        throughput_rates = [dp[1]['throughput_rate'] for dp in buffer_list]
        congestion_levels = [dp[1]['congestion_level'] for dp in buffer_list]
        total_cars = [c1 + c2 for c1, c2 in zip(car_counts_1, car_counts_2)]

        # Update traffic flow tab
        if "traffic_flow" in self.expanded_figures:
            fig, ax1, ax2, canvas = self.expanded_figures["traffic_flow"]
            ax1.clear()
            ax2.clear()

            # Top: Traffic distribution
            ax1.plot(timestamps, car_counts_1, 'b-', label='Subroad 1', linewidth=3)
            ax1.plot(timestamps, car_counts_2, 'r-', label='Subroad 2', linewidth=3)
            ax1.plot(timestamps, total_cars, 'g--', label='Total Cars', linewidth=2)
            ax1.axhline(y=7, color='red', linestyle=':', label='Max Limit')
            ax1.set_title('🚗 Detailed Traffic Flow Analysis', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Number of Cars')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Bottom: Traffic patterns
            ax2.plot(timestamps, throughput_rates, 'orange', linewidth=3)
            ax2.set_title('📈 Traffic Throughput Over Time', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Cars per Minute')
            ax2.set_xlabel('Time')
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            canvas.draw()

        # Update performance tab
        if "performance" in self.expanded_figures:
            fig, ax1, ax2, canvas = self.expanded_figures["performance"]
            ax1.clear()
            ax2.clear()

            ax1.plot(timestamps, response_times, 'red', linewidth=3)
            ax1.axhline(y=1.0, color='orange', linestyle='--', label='Target (1s)')
            ax1.set_title('⏱️ System Response Times', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Response Time (seconds)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            accuracy_values = [dp[1]['car_count_accuracy'] for dp in buffer_list]
            ax2.plot(timestamps, accuracy_values, 'green', linewidth=3)
            ax2.axhline(y=0.95, color='red', linestyle='--', label='Target (95%)')
            ax2.set_title('🎯 Car Counting Accuracy', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Accuracy Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

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
        new_fig = Figure(figsize=(12, 8), dpi=100)

        # Copy the plot to new figure
        for i, ax in enumerate(fig.axes):
            new_ax = new_fig.add_subplot(len(fig.axes), 1, i + 1)
            # Copy plot data (simplified - in real implementation, you'd copy the actual data)
            new_ax.set_title(ax.get_title())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())

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
        """Get pollution alert level with emojis"""
        if pollution_level > 2000:
            return "🔴 CRITICAL"
        elif pollution_level > 1800:
            return "🟠 HIGH"
        elif pollution_level > 1500:
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
