#!/usr/bin/env python3
"""
VRPTW Main Script
Solves Vehicle Routing Problem with Time Windows for vegetable delivery.

This script:
1. Reads and parses CSV files from csv_data/input/
2. Analyzes and preprocesses input data
3. Constructs and solves VRPTW problems
4. Supports multiple solving algorithms
5. Outputs results to csv_data/output/result.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Customer:
    """Customer data structure"""
    id: str
    name: str
    address: str
    latitude: float
    longitude: float
    time_window_start: str
    time_window_end: str
    demand: Dict[str, float]  # product_id -> quantity
    priority: int = 1

@dataclass
class Vehicle:
    """Vehicle data structure"""
    id: str
    capacity_weight: float
    capacity_volume: float
    start_location: str
    end_location: str
    available_time_start: str
    available_time_end: str
    cost_per_km: float = 1.0

@dataclass
class Product:
    """Product data structure"""
    id: str
    name: str
    weight_per_unit: float
    volume_per_unit: float
    category: str

@dataclass
class Location:
    """Location data structure"""
    id: str
    name: str
    address: str
    latitude: float
    longitude: float
    location_type: str  # 'warehouse', 'customer', etc.

class DataManager:
    """Handles reading and parsing of CSV input data"""
    
    def __init__(self, input_dir: str = "csv_data/input"):
        self.input_dir = Path(input_dir)
        self.customers: List[Customer] = []
        self.vehicles: List[Vehicle] = []
        self.products: List[Product] = []
        self.locations: List[Location] = []
        
    def load_all_data(self) -> bool:
        """Load all CSV data files"""
        try:
            logger.info("Loading data from CSV files...")
            
            # Create input directory if it doesn't exist
            os.makedirs(self.input_dir, exist_ok=True)
            
            # Load each type of data
            self._load_customers()
            self._load_vehicles()
            self._load_products()
            self._load_locations()
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"  - {len(self.customers)} customers")
            logger.info(f"  - {len(self.vehicles)} vehicles")
            logger.info(f"  - {len(self.products)} products")
            logger.info(f"  - {len(self.locations)} locations")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _load_customers(self):
        """Load customer data from CSV"""
        # Load from 需排线销售订单.csv (main customer orders)
        csv_files = list(self.input_dir.glob("*需排线销售订单*.csv")) or list(self.input_dir.glob("*customer*.csv"))
        
        # Also load time windows from separate file
        time_window_files = list(self.input_dir.glob("*需排线销售订单带计划发货时间*.csv"))
        time_windows = {}
        
        # Load time windows first
        for csv_file in time_window_files:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                logger.info(f"Loading time windows from {csv_file.name}")
                for _, row in df.iterrows():
                    order_id = str(row['销售订单'])
                    time_windows[order_id] = {
                        'start': str(row['计划收货开始时间']),
                        'end': str(row['计划收货结束时间'])
                    }
            except Exception as e:
                logger.warning(f"Could not load time windows from {csv_file}: {e}")
        
        # Load main customer data
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                logger.info(f"Loading customers from {csv_file.name}")
                
                for _, row in df.iterrows():
                    try:
                        # Check if required columns exist and have valid data
                        required_columns = ['销售订单', '子客户编码', '子客户名称', '子客户地址']
                        if not all(col in row.index for col in required_columns):
                            continue
                        
                        # Skip rows with missing critical data
                        if pd.isna(row['销售订单']) or pd.isna(row['子客户编码']):
                            continue
                            
                        order_id = str(row['销售订单']).strip()
                        customer_id = str(row['子客户编码']).strip()
                        customer_name = str(row['子客户名称']).strip() if pd.notna(row['子客户名称']) else f"Customer_{customer_id}"
                        address = str(row['子客户地址']).strip() if pd.notna(row['子客户地址']) else "Unknown Address"
                        
                        # Skip if essential data is empty
                        if not order_id or not customer_id or order_id == 'nan' or customer_id == 'nan':
                            continue
                        
                        # Get coordinates with better error handling
                        lat = 0.0
                        lng = 0.0
                        if '经纬度' in row.index and pd.notna(row['经纬度']):
                            try:
                                lat = float(row['经纬度'])
                            except (ValueError, TypeError):
                                lat = 0.0
                        
                        if '经纬度1' in row.index and pd.notna(row['经纬度1']):
                            try:
                                lng = float(row['经纬度1'])
                            except (ValueError, TypeError):
                                lng = 0.0
                        
                        # Get time windows for this order
                        tw = time_windows.get(order_id, {'start': '06:00:00', 'end': '18:00:00'})
                        
                        customer = Customer(
                            id=customer_id,
                            name=customer_name,
                            address=address,
                            latitude=lat,
                            longitude=lng,
                            time_window_start=tw['start'],
                            time_window_end=tw['end'],
                            demand={}  # Will be filled from order details
                        )
                        
                        self.customers.append(customer)
                        
                    except Exception as e:
                        # Only log detailed errors for debugging, not every row issue
                        if len(str(e)) > 20:  # Only log substantial errors
                            logger.debug(f"Error processing customer row: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Could not load customers from {csv_file}: {e}")
    
    def _load_vehicles(self):
        """Load vehicle data from CSV"""
        # Note: 车辆商品池.csv seems to have formatting issues, we'll create default vehicles for now
        logger.info("Creating default vehicle fleet (车辆商品池.csv has formatting issues)")
        
        # Create a default fleet of vehicles with different capacities
        default_vehicles = [
            {'id': 'V001', 'weight_capacity': 1000, 'volume_capacity': 15, 'cost_per_km': 1.2},
            {'id': 'V002', 'weight_capacity': 1500, 'volume_capacity': 20, 'cost_per_km': 1.5},
            {'id': 'V003', 'weight_capacity': 2000, 'volume_capacity': 25, 'cost_per_km': 1.8},
            {'id': 'V004', 'weight_capacity': 2500, 'volume_capacity': 30, 'cost_per_km': 2.0},
            {'id': 'V005', 'weight_capacity': 3000, 'volume_capacity': 35, 'cost_per_km': 2.2},
        ]
        
        for vehicle_data in default_vehicles:
            vehicle = Vehicle(
                id=vehicle_data['id'],
                capacity_weight=vehicle_data['weight_capacity'],
                capacity_volume=vehicle_data['volume_capacity'],
                start_location='warehouse',
                end_location='warehouse',
                available_time_start='06:00:00',
                available_time_end='20:00:00',
                cost_per_km=vehicle_data['cost_per_km']
            )
            self.vehicles.append(vehicle)
    
    def _load_products(self):
        """Load product data from CSV"""
        # Load from 修正商品体积.csv and 订单商品明细.csv
        volume_files = list(self.input_dir.glob("*修正商品体积*.csv"))
        detail_files = list(self.input_dir.glob("*订单商品明细*.csv"))
        
        # First load volume corrections
        volume_corrections = {}
        for csv_file in volume_files:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                logger.info(f"Loading product volumes from {csv_file.name}")
                for _, row in df.iterrows():
                    product_id = str(row['商品编码'])
                    volume = float(row['修正后规格体积L']) if pd.notna(row['修正后规格体积L']) else 0.0
                    volume_corrections[product_id] = volume
            except Exception as e:
                logger.warning(f"Could not load product volumes from {csv_file}: {e}")
        
        # Load product details
        products_dict = {}
        for csv_file in detail_files:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                logger.info(f"Loading products from {csv_file.name}")
                
                for _, row in df.iterrows():
                    try:
                        product_id = str(row['商品编码'])
                        product_name = str(row['商品名称'])
                        unit = str(row['单位'])
                        
                        # Calculate volume per unit
                        volume_per_unit = 0.0
                        if product_id in volume_corrections:
                            volume_per_unit = volume_corrections[product_id]
                        elif pd.notna(row['规格体积L']):
                            volume_per_unit = float(row['规格体积L'])
                        elif pd.notna(row['计算规则（L/kg)']):
                            volume_per_unit = float(row['计算规则（L/kg)'])
                        
                        # Estimate weight per unit (placeholder - could be improved)
                        weight_per_unit = volume_per_unit * 0.8 if volume_per_unit > 0 else 1.0
                        
                        if product_id not in products_dict:
                            product = Product(
                                id=product_id,
                                name=product_name,
                                weight_per_unit=weight_per_unit,
                                volume_per_unit=volume_per_unit,
                                category=unit
                            )
                            products_dict[product_id] = product
                            
                    except Exception as e:
                        logger.warning(f"Error processing product row: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Could not load products from {csv_file}: {e}")
        
        self.products = list(products_dict.values())
    
    def _load_locations(self):
        """Load location data from CSV"""
        # Load warehouse location from 出库仓库.csv
        warehouse_files = list(self.input_dir.glob("*出库仓库*.csv"))
        
        for csv_file in warehouse_files:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                logger.info(f"Loading warehouse location from {csv_file.name}")
                
                # Extract warehouse info from the specific format
                warehouse_info = df.iloc[0, 1] if len(df) > 0 else "重庆市沙坪坝区"
                coords = df.iloc[1, 1:3] if len(df) > 1 else [106.384825, 29.676384]
                
                warehouse = Location(
                    id='warehouse',
                    name='彩食鲜供应链仓库',
                    address=str(warehouse_info),
                    latitude=float(coords.iloc[1]) if len(coords) > 1 else 29.676384,
                    longitude=float(coords.iloc[0]) if len(coords) > 0 else 106.384825,
                    location_type='warehouse'
                )
                
                self.locations.append(warehouse)
                
            except Exception as e:
                logger.warning(f"Could not load warehouse location from {csv_file}: {e}")
        
        # Add customer locations from customer data
        for customer in self.customers:
            location = Location(
                id=f"customer_{customer.id}",
                name=customer.name,
                address=customer.address,
                latitude=customer.latitude,
                longitude=customer.longitude,
                location_type='customer'
            )
            self.locations.append(location)

class DataPreprocessor:
    """Handles data analysis and preprocessing"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        
    def analyze_data(self) -> Dict[str, Any]:
        """Analyze input data and return summary statistics"""
        analysis = {
            'customers_count': len(self.data_manager.customers),
            'vehicles_count': len(self.data_manager.vehicles),
            'products_count': len(self.data_manager.products),
            'locations_count': len(self.data_manager.locations),
            'total_demand': {},
            'time_windows_analysis': {},
            'capacity_analysis': {}
        }
        
        logger.info("Analyzing input data...")
        
        # Add more detailed analysis here
        # This will be expanded based on actual data structure
        
        return analysis
    
    def preprocess_data(self) -> bool:
        """Preprocess data for VRPTW solving"""
        try:
            logger.info("Preprocessing data...")
            
            # Data validation
            self._validate_data()
            
            # Data cleaning and normalization
            self._clean_data()
            
            # Generate derived data
            self._generate_derived_data()
            
            logger.info("Data preprocessing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            return False
    
    def _validate_data(self):
        """Validate input data consistency"""
        # Add validation logic here
        pass
    
    def _clean_data(self):
        """Clean and normalize data"""
        # Add data cleaning logic here
        pass
    
    def _generate_derived_data(self):
        """Generate derived data needed for VRPTW solving"""
        # Add derived data generation logic here
        pass

class VRPTWProblem:
    """VRPTW problem representation and construction"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.distance_matrix: Optional[np.ndarray] = None
        self.time_matrix: Optional[np.ndarray] = None
        self.volume_calculator = None  # Will be set by user-provided API
        
    def construct_problem(self) -> bool:
        """Construct the VRPTW problem from input data"""
        try:
            logger.info("Constructing VRPTW problem...")
            
            # Build distance and time matrices (placeholder for user APIs)
            self._build_distance_matrix()
            self._build_time_matrix()
            
            # Validate problem construction
            self._validate_problem()
            
            logger.info("VRPTW problem constructed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error constructing VRPTW problem: {e}")
            return False
    
    def _build_distance_matrix(self):
        """Build distance matrix using user-provided API"""
        # Placeholder - will use user-provided distance API
        logger.info("Building distance matrix (placeholder)")
        pass
    
    def _build_time_matrix(self):
        """Build time matrix using user-provided API"""
        # Placeholder - will use user-provided time API
        logger.info("Building time matrix (placeholder)")
        pass
    
    def _validate_problem(self):
        """Validate the constructed problem"""
        # Add problem validation logic
        pass

class VRPTWSolver:
    """Base class for VRPTW solving algorithms"""
    
    def __init__(self, problem: VRPTWProblem):
        self.problem = problem
        self.solution = None
        
    def solve(self) -> Dict[str, Any]:
        """Solve the VRPTW problem - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement solve method")

class GreedyVRPTWSolver(VRPTWSolver):
    """Greedy algorithm for VRPTW"""
    
    def solve(self) -> Dict[str, Any]:
        """Solve using greedy algorithm"""
        logger.info("Solving VRPTW using Greedy algorithm...")
        
        # Placeholder for greedy algorithm implementation
        solution = {
            'algorithm': 'Greedy',
            'routes': [],
            'total_distance': 0,
            'total_time': 0,
            'vehicles_used': 0,
            'status': 'solved'
        }
        
        self.solution = solution
        return solution

class GeneticVRPTWSolver(VRPTWSolver):
    """Genetic algorithm for VRPTW"""
    
    def solve(self) -> Dict[str, Any]:
        """Solve using genetic algorithm"""
        logger.info("Solving VRPTW using Genetic algorithm...")
        
        # Placeholder for genetic algorithm implementation
        solution = {
            'algorithm': 'Genetic',
            'routes': [],
            'total_distance': 0,
            'total_time': 0,
            'vehicles_used': 0,
            'status': 'solved'
        }
        
        self.solution = solution
        return solution

class OutputManager:
    """Handles output generation and formatting"""
    
    def __init__(self, output_dir: str = "csv_data/output"):
        self.output_dir = Path(output_dir)
        
    def generate_output(self, solution: Dict[str, Any], filename: str = "result.csv") -> bool:
        """Generate output CSV file"""
        try:
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            output_path = self.output_dir / filename
            
            logger.info(f"Generating output to {output_path}")
            
            # Convert solution to DataFrame format
            output_df = self._format_solution_to_dataframe(solution)
            
            # Save to CSV
            output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"Output generated successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating output: {e}")
            return False
    
    def _format_solution_to_dataframe(self, solution: Dict[str, Any]) -> pd.DataFrame:
        """Format solution data to DataFrame"""
        # Placeholder - format based on required output structure
        data = {
            'Vehicle_ID': [],
            'Route': [],
            'Customer_Sequence': [],
            'Total_Distance': [],
            'Total_Time': [],
            'Load_Utilization': []
        }
        
        # This will be expanded based on actual solution structure
        return pd.DataFrame(data)

class VRPTWMain:
    """Main class orchestrating the entire VRPTW solving process"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.preprocessor = None
        self.problem = None
        self.solvers = {
            'greedy': GreedyVRPTWSolver,
            'genetic': GeneticVRPTWSolver
        }
        self.output_manager = OutputManager()
        
    def run(self, algorithm: str = 'greedy') -> bool:
        """Run the complete VRPTW solving pipeline"""
        try:
            logger.info("Starting VRPTW solving pipeline...")
            
            # Step 1: Load and parse data
            if not self.data_manager.load_all_data():
                logger.error("Failed to load data")
                return False
            
            # Step 2: Analyze and preprocess data
            self.preprocessor = DataPreprocessor(self.data_manager)
            analysis = self.preprocessor.analyze_data()
            logger.info(f"Data analysis: {analysis}")
            
            if not self.preprocessor.preprocess_data():
                logger.error("Failed to preprocess data")
                return False
            
            # Step 3: Construct VRPTW problem
            self.problem = VRPTWProblem(self.data_manager)
            if not self.problem.construct_problem():
                logger.error("Failed to construct VRPTW problem")
                return False
            
            # Step 4: Solve the problem
            if algorithm not in self.solvers:
                logger.error(f"Unknown algorithm: {algorithm}")
                return False
            
            solver_class = self.solvers[algorithm]
            solver = solver_class(self.problem)
            solution = solver.solve()
            
            # Step 5: Generate output
            if not self.output_manager.generate_output(solution):
                logger.error("Failed to generate output")
                return False
            
            logger.info("VRPTW solving pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in VRPTW pipeline: {e}")
            return False
    
    def set_volume_calculator_api(self, api_function):
        """Set user-provided volume calculation API"""
        if self.problem:
            self.problem.volume_calculator = api_function
            logger.info("Volume calculator API set successfully")
    
    def set_distance_time_api(self, distance_api, time_api):
        """Set user-provided distance and time calculation APIs"""
        if self.problem:
            self.problem.distance_api = distance_api
            self.problem.time_api = time_api
            logger.info("Distance and time APIs set successfully")

def main():
    """Main entry point"""
    print("VRPTW Solver - Vehicle Routing Problem with Time Windows")
    print("=" * 60)
    
    # Initialize main solver
    vrptw_main = VRPTWMain()
    
    # Run with default greedy algorithm
    success = vrptw_main.run(algorithm='greedy')
    
    if success:
        print("\n✅ VRPTW solving completed successfully!")
        print("Check csv_data/output/result.csv for results")
    else:
        print("\n❌ VRPTW solving failed. Check logs for details.")

if __name__ == "__main__":
    main()
