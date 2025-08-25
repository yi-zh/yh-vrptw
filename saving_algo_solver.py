import copy
import json
import re
import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from datetime import datetime
import traceback
import random
from copy import deepcopy
import pandas as pd
from pathlib import Path

# 继承已有数据类
from main import Customer, Vehicle, Product, Location, VRPTWSolver, VRPTWProblem, logger, DataManager, OutputManager, \
    DataPreprocessor


VEHICLE_COST_MAP = {
    0: '0-20KM（元）',
    1: '21-40KM（元）',
    2: '41-60KM（元）',
    3: '61-80KM（元）',
    4: '81-100KM（元）',
    5: '100公里以上（元/KM）'
}

_routes_matrix_cache = None
_routes_matrix_42_cache = None
_distance_cache = {}
_travel_time_cache = {}

def load_routes_matrix():
    """Load routes matrix from CSV file and cache it"""
    global _routes_matrix_cache, _routes_matrix_42_cache
    
    if _routes_matrix_cache is not None and _routes_matrix_42_cache is not None:
        return _routes_matrix_cache, _routes_matrix_42_cache
    
    # Load regular routes matrix
    regular_matrix = {}
    matrix_42 = {}
    
    try:
        # Load regular routes matrix
        routes_file = Path("csv_data/input/route_matrix.csv")
        if routes_file.exists():
            print(f"Loading regular routes matrix from {routes_file.name}...")
            regular_matrix = _load_single_matrix(routes_file)
        else:
            print(f"Regular routes matrix file not found: {routes_file}")
            
        # Load 4.2m vehicle routes matrix
        routes_42_file = Path("csv_data/input/route_matrix_42.csv")
        if routes_42_file.exists():
            print(f"Loading 4.2m vehicle routes matrix from {routes_42_file.name}...")
            matrix_42 = _load_single_matrix(routes_42_file)
        else:
            print(f"4.2m routes matrix file not found: {routes_42_file}")
            # If 4.2m matrix doesn't exist, use regular matrix as fallback
            matrix_42 = regular_matrix
            
    except Exception as e:
        print(f"Error loading routes matrices: {e}")
    
    # Cache the matrices
    _routes_matrix_cache = regular_matrix
    _routes_matrix_42_cache = matrix_42
    
    return regular_matrix, matrix_42

def _load_single_matrix(file_path):
    """Load a single routes matrix file"""
    try:
        # Read the routes matrix CSV
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # Print column info for debugging
        print(f"Routes matrix columns: {list(df.columns)}")
        print(f"Routes matrix shape: {df.shape}")
        
        # Create a dictionary for fast lookup
        routes_dict = {}
        
        # Try to identify the correct column names
        origin_col = None
        dest_col = None
        distance_col = None
        duration_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'origin' in col_lower or 'from' in col_lower or 'start' in col_lower:
                origin_col = col
            elif 'destination' in col_lower or 'dest' in col_lower:
                dest_col = col
            elif 'distance' in col_lower or 'km' in col_lower or 'meter' in col_lower:
                distance_col = col
            elif 'duration' in col_lower or 'time' in col_lower or 'minute' in col_lower:
                duration_col = col
        
        if origin_col and dest_col and (distance_col or duration_col):
            print(f"Using columns: origin={origin_col}, dest={dest_col}, distance={distance_col}, duration={duration_col}")
            
            for _, row in df.iterrows():
                try:
                    origin = str(row[origin_col]).strip()
                    dest = str(row[dest_col]).strip()
                    
                    distance = float(row[distance_col]) if distance_col and pd.notna(row[distance_col]) else None
                    duration = float(row[duration_col]) if duration_col and pd.notna(row[duration_col]) else None
                    
                    # Store both directions
                    routes_dict[(origin, dest)] = {
                        'distance': distance,
                        'duration': duration
                    }
                    routes_dict[(dest, origin)] = {
                        'distance': distance,
                        'duration': duration
                    }
                    
                except (ValueError, KeyError) as e:
                    continue
            
            print(f"Loaded {len(routes_dict)} route entries from {file_path.name}")
            return routes_dict
        else:
            print(f"Could not identify required columns in routes matrix")
            print(f"Available columns: {list(df.columns)}")
            
    except Exception as e:
        print(f"Error loading routes matrix from {file_path}: {e}")
    
    return {}

def get_location_id(location) -> str:
    """Extract location ID from location object"""
    if hasattr(location, 'id'):
        return str(location.id)
    elif hasattr(location, 'name'):
        return str(location.name)
    elif hasattr(location, 'sub_customer_code'):
        return str(location.sub_customer_code)
    else:
        return str(location)

def get_appropriate_matrix(vehicle_type=None):
    """Get the appropriate routes matrix based on vehicle type"""
    regular_matrix, matrix_42 = load_routes_matrix()
    
    # Use 4.2m matrix for 4.2m vehicles, regular matrix for others
    if vehicle_type and "4.2" in str(vehicle_type):
        return matrix_42
    else:
        return regular_matrix

def calculate_distance(loc1, loc2, vehicle_type=None) -> float:
    """计算两个位置之间的距离（使用路线矩阵数据）"""
    global _distance_cache
    
    # Get location IDs
    loc1_id = get_location_id(loc1)
    loc2_id = get_location_id(loc2)
    
    # Create cache key including vehicle type
    cache_key = (','.join([str(loc1.longitude),str(loc1.latitude)]),','.join([str(loc2.longitude),str(loc2.latitude)]), str(vehicle_type) if vehicle_type else "default")
    
    if cache_key in _distance_cache:
        return _distance_cache[cache_key]
    
    # Get appropriate routes matrix based on vehicle type
    routes_matrix = get_appropriate_matrix(vehicle_type)
    
    # Create lookup key for routes matrix
    matrix_key = (','.join([str(loc1.longitude),str(loc1.latitude)]),','.join([str(loc2.longitude),str(loc2.latitude)]))
    
    # Try to find distance in routes matrix
    if matrix_key in routes_matrix and routes_matrix[matrix_key]['distance'] is not None:
        distance = routes_matrix[matrix_key]['distance']
        _distance_cache[cache_key] = distance/1000.0
        return distance/1000.0
    
    # Fallback to euclidean distance calculation
    try:
        if hasattr(loc1, 'latitude') and hasattr(loc1, 'longitude') and \
           hasattr(loc2, 'latitude') and hasattr(loc2, 'longitude'):
            # Use Haversine formula for more accurate distance calculation
            lat1, lon1 = float(loc1.latitude), float(loc1.longitude)
            lat2, lon2 = float(loc2.latitude), float(loc2.longitude)
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371  # Earth's radius in kilometers
            distance = c * r/1000.0
        else:
            # Simple euclidean distance as last resort
            distance = math.hypot(loc1.longitude - loc2.longitude, loc1.latitude - loc2.latitude) * 100
        
        _distance_cache[cache_key] = distance
        return distance
        
    except Exception as e:
        print(f"Error calculating distance between {loc1_id} and {loc2_id}: {e}")
        # Return a default distance
        default_distance = 5.0
        _distance_cache[cache_key] = default_distance
        return default_distance

def calculate_travel_time(distance: float = None, loc1=None, loc2=None, vehicle_type=None) -> float:
    """根据距离或路线矩阵计算旅行时间（分钟）"""
    global _travel_time_cache
    
    # If locations are provided, try to use routes matrix first
    if loc1 is not None and loc2 is not None:
        loc1_id = get_location_id(loc1)
        loc2_id = get_location_id(loc2)
        
        # Create cache key including vehicle type
        cache_key = (
            ','.join([str(loc1.longitude), str(loc1.latitude)]), 
            ','.join([str(loc2.longitude), str(loc2.latitude)]),
            str(vehicle_type) if vehicle_type else "default"
        )

        if cache_key in _travel_time_cache:
            return _travel_time_cache[cache_key]
        
        # Get appropriate routes matrix based on vehicle type
        routes_matrix = get_appropriate_matrix(vehicle_type)
        
        # Create lookup key for routes matrix
        matrix_key = (
            ','.join([str(loc1.longitude), str(loc1.latitude)]), 
            ','.join([str(loc2.longitude), str(loc2.latitude)])
        )
        
        # Try to find travel time in routes matrix
        if matrix_key in routes_matrix and routes_matrix[matrix_key]['duration'] is not None:
            travel_time = routes_matrix[matrix_key]['duration']
            _travel_time_cache[cache_key] = travel_time/60.0
            return travel_time/60.0

    if distance == 0.0:
        return 0.0

    # Fallback to distance-based calculation
    if distance is None and loc1 is not None and loc2 is not None:
        distance = calculate_distance(loc1, loc2, vehicle_type)
    
    if distance is not None:
        avg_speed = 40.0  # 平均速度，公里/小时
        travel_time = (distance / avg_speed) * 60  # 转换为分钟
        
        # Cache the result if we have location IDs
        if loc1 is not None and loc2 is not None:
            cache_key = (
                ','.join([str(loc1.longitude), str(loc1.latitude)]), 
                ','.join([str(loc2.longitude), str(loc2.latitude)]),
                str(vehicle_type) if vehicle_type else "default"
            )
            _travel_time_cache[cache_key] = travel_time
        
        return travel_time
    
    # Default travel time if all else fails
    return 30.0  # 30 minutes default

def extract_district(address: str) -> str:
    """
    从地址中提取区名（仅返回XX区格式的结果）
    """
    # 匹配2-5个汉字加"区"的模式，覆盖大多数区名
    pattern = r'([\u4e00-\u9fa5]{2}区)'
    match = re.search(pattern, address)

    if match:
        return match.group(1)
    return ""  # 未找到区名时返回空字符串


def get_service_time(customer: Customer) -> float:
    """获取客户的服务时间（分钟）"""
    addition_time = customer.extra_work_hours * 60
    volume = customer.volume
    delivery_method = customer.delivery_method if (customer.delivery_method is not None
                                                   or customer.delivery_method != ""
                                                   or len(customer.delivery_method) > 0) else "称重点数"
    if delivery_method == "信任交接":
        if volume <= 500:
            return 15 + addition_time
        else:
            return 15 + (volume-500) / 1000 * 10 + addition_time
    else:
    # elif delivery_method == "称重点数":
        if volume <= 500:
            return 20 + volume / 1000 * 15 + addition_time
        else:
            return 20 + (volume-500) / 1000 * 10 + volume / 1000 * 15 + addition_time
    # else:
    #     raise ValueError(f"Order {customer.id} 的交接方式错误。its way is {customer.delivery_method}")


# def calculate_distance(loc1, loc2) -> float:
#     """计算两个位置之间的距离（使用已有方法或实现）"""
#     # 这里可以使用实际的距离计算方法
#     # 示例使用欧氏距离
#     return math.hypot(loc1.longitude - loc2.longitude, loc1.latitude - loc2.latitude) * 100  # 简单转换为公里


# def calculate_travel_time(distance: float) -> float:
#     """根据距离计算旅行时间（分钟）"""
#     avg_speed = 40.0  # 平均速度，公里/小时
#     return (distance / avg_speed) * 60  # 转换为分钟


def parse_time(time_str: str) -> float:
    """将时间字符串转换为分钟数（自午夜起）"""
    try:
        # 看PPT是当天排线，不涉及跨天
        time_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return time_obj.hour * 60 + time_obj.minute
    except ValueError:
        try:
            # 适配 "07:15:00" 格式
            time_obj = datetime.strptime(time_str, "%H:%M:%S")
            return time_obj.hour * 60 + time_obj.minute
        except ValueError:
            try:
                # 适配 "07:15" 格式
                time_obj = datetime.strptime(time_str, "%H:%M")
                return time_obj.hour * 60 + time_obj.minute
            except ValueError:
                return 0.0  # 所有格式都不匹配时返回默认值


def get_cost_stage(distance: float):
    if distance < 21:
        return 0
    elif distance < 41:
        return 1
    elif distance < 61:
        return 2
    elif distance < 81:
        return 3
    elif distance <= 100:
        return 4
    else:
        return 5


def calculate_transportation_cost(route:Dict[str, Any], fee_map):
    selected_vehicle_type = route['vehicle_type']
    distance = route['total_distance']
    customers_num = len(route['customers'])

    base_fee = 0
    cost_stage = get_cost_stage(distance)
    if cost_stage <= 4:
        base_fee = fee_map[selected_vehicle_type][VEHICLE_COST_MAP[cost_stage]]
    else:
        base_fee = fee_map[selected_vehicle_type][VEHICLE_COST_MAP[4]]
        if selected_vehicle_type in ["大型面包车", "4.2m厢式货车"]:
            base_fee += float(fee_map[selected_vehicle_type][VEHICLE_COST_MAP[5]]) * (
                        distance - 100)

    if selected_vehicle_type == "4.2m厢式货车":
        return base_fee + 20 * max(0, customers_num - 2)
    else:
        return base_fee + 15 * max(0, customers_num - 2)


def calculate_customer_load(customer: Customer, product_map):
    """计算客的总重量和总体积需求"""
    total_weight = 0.0
    total_volume = 0.0

    for product_id, quantity in customer.demand.items():
        product = product_map[product_id]
        if product:
            if product.category == 'KG':
                total_weight += quantity
                total_volume += product.volume_per_unit * quantity
            else:
                total_weight += product.weight_per_unit * quantity
                total_volume += product.volume_per_unit * quantity

    return {'weight': total_weight, 'volume': total_volume}


class SavingsAlgorithmSolver(VRPTWSolver):
    """带时间窗的节约算法求解VRPTW问题"""

    def __init__(self, problem: VRPTWProblem, is_split:bool):
        super().__init__(problem)
        self.savings = []  # 存储节约量
        self.routes = []  # 存储路径
        self.total_cost = 0.0
        self.is_split = is_split

    def solve(self) -> Dict[str, Any]:
        """实现带时间窗的节约算法"""
        logger.info("搜索VRPTW问题初始解...")

        try:
            self._establish_info_map()
            self.problem.data_manager.vehicles = sorted(self.problem.data_manager.vehicles, key=lambda x: x.capacity_volume)
            self.max_volume_one_vehicle = self.problem.data_manager.vehicles[-1].capacity_volume * 0.85
            # 1. 初始化：为每个客户创建单独路径
            self._initialize_routes()

            # 2. 计算所有客户对之间的节约量
            # 同一个主客户下的不同子客户由于地址相同，讲道理在saving的时候就应该放在一条车线上了
            self._calculate_savings()

            # 3. 按节约量排序
            self._sort_savings()

            # 4. 合并路径
            self._merge_routes()

            # 5. 计算总成本和其他指标
            self._calculate_metrics()

            solution = {
                'algorithm': 'Savings with Time Windows',
                'routes': self.routes,
                'total_cost': self.total_cost,
                'vehicles_used': len(self.routes),
                'status': 'solved'
            }

            self.solution = solution
            return solution

        except Exception as e:
            logger.error(f"节约算法求解出错: {e}")
            tb_str = traceback.format_exc()
            print(tb_str)  # 打印字符串形式的堆栈信息
            return {
                'algorithm': 'Savings with Time Windows',
                'routes': [],
                'total_cost': 0,
                'status': 'failed',
                'error': str(e)
            }

    def _establish_info_map(self):
        self.vehicle_map = {}
        self.customer_map = {}
        self.product_map = {}
        for vehicle in self.problem.data_manager.vehicles:
            self.vehicle_map[vehicle.id] = vehicle
        for customer in self.problem.data_manager.customers:
            self.customer_map[customer.id] = customer
        for product in self.problem.data_manager.products:
            self.product_map[product.id] = product

    def _initialize_routes(self):
        """初始化路径：每个客户单独一条路径（仓库-客户-仓库）"""
        warehouse = self._get_warehouse_location()

        for customer in self.problem.data_manager.customers:
            # 计算客户需求
            load = self._calculate_customer_load(customer)
            customer.volume = load['volume']
            customer.weight = load['weight']
            district = extract_district(customer.address)

            # appropriate_vehicle_type = self._find_suitable_vehicle(load['weight'], load['volume']).vehicle_type \
            #     if customer.vehicle_restriction == "" else None
            # specified_vehicle_type = customer.vehicle_restriction if customer.vehicle_restriction != "" else None

            # 计算往返距离
            to_customer = calculate_distance(warehouse, customer)
            return_distance = calculate_distance(customer, warehouse)

            # 计算时间
            travel_time = calculate_travel_time(to_customer, warehouse, customer)
            service_time = get_service_time(customer)
            return_time = calculate_travel_time(return_distance, customer, warehouse)

            # 时间窗处理
            tw_start = parse_time(customer.time_window_start)
            tw_end = parse_time(customer.time_window_end)
            arrival_time = max(travel_time, tw_start)
            departure_time = arrival_time + service_time

            route = {
                'vehicle_id': None,  # 尚未分配车辆
                'vehicle_type': None,
                'customers': [customer.id],
                'sequence': [warehouse.id, customer.id, warehouse.id],
                'total_distance': to_customer + return_distance,
                'total_time': travel_time + service_time + return_time,
                'load_weight': load['weight'],
                'load_volume': load['volume'],
                'arrival_times': {},
                'departure_times': {},
                'district': {district},
                'specified_vehicle': None,
                'height_restricted': customer.height_restricted,
                'breakpoint': None,
                'single_vehicle': customer.delivery_type == "单点配送",
                "vehicle_work_start_time": arrival_time - travel_time,
                "vehicle_work_end_time": departure_time + return_time,
            }

            if self.is_split and load['volume'] > self.max_volume_one_vehicle:
                if load['volume'] % self.max_volume_one_vehicle == 0:
                    vehicle_num = int(load['volume'] / self.max_volume_one_vehicle)
                else:
                    vehicle_num = int(load['volume'] / self.max_volume_one_vehicle) + 1
                for i in range(vehicle_num-1):
                    one_route = copy.deepcopy(route)
                    new_customer_id = f"{route['customers'][0]}-part{i}"
                    one_route['customers'][0] = new_customer_id
                    one_route['sequence'][1] = new_customer_id
                    one_route['load_weight'] = route['load_weight'] / vehicle_num
                    one_route['load_volume'] = self.max_volume_one_vehicle
                    one_route['arrival_times'][new_customer_id] = arrival_time
                    one_route['departure_times'][new_customer_id] = departure_time
                    self.routes.append(one_route)
                    self._add_virtual_customer(new_customer_id, self.max_volume_one_vehicle, route['load_weight'] / vehicle_num, customer)
                last_one_route = copy.deepcopy(route)
                new_customer_id = f"{route['customers'][0]}-part{vehicle_num-1}"
                last_one_route['customers'][0] = new_customer_id
                last_one_route['sequence'][1] = new_customer_id
                last_one_route['load_weight'] = route['load_weight'] / vehicle_num
                last_one_route['load_volume'] = load['volume'] - (vehicle_num-1) * self.max_volume_one_vehicle
                last_one_route['arrival_times'][new_customer_id] = arrival_time
                last_one_route['departure_times'][new_customer_id] = departure_time
                self.routes.append(last_one_route)
                self._add_virtual_customer(new_customer_id, load['volume'] - (vehicle_num-1) * self.max_volume_one_vehicle,
                                           route['load_weight'] / vehicle_num, customer)
                del self.customer_map[customer.id]
            else:
                route['arrival_times'][customer.id] = arrival_time
                route['departure_times'][customer.id] = departure_time
                self.routes.append(route)

    def _add_virtual_customer(self, cus_id, volume, weight, original_customer):
        virtual_customer = copy.deepcopy(original_customer)
        virtual_customer.id = cus_id
        virtual_customer.volume = volume
        virtual_customer.weight = weight
        self.customer_map[cus_id] = virtual_customer

    def _calculate_savings(self):
        """计算所有客户对之间的节约量"""
        warehouse = self._get_warehouse_location()

        for i, customer_i in enumerate(self.problem.data_manager.customers):
            if customer_i.delivery_type == "单点配送":
                continue
            for j, customer_j in enumerate(self.problem.data_manager.customers):
                if customer_j.delivery_type == "单点配送":
                    continue
                if i >= j:
                    continue  # 避免重复计算

                c0i = calculate_distance(customer_i, warehouse)
                cj0 = calculate_distance(warehouse, customer_j)
                cij = calculate_distance(customer_i, customer_j)

                saving = c0i + cj0 - cij

                if saving > 0:  # 只保留正的节约量
                    self.savings.append({
                        'i': customer_i.id,
                        'j': customer_j.id,
                        'saving': saving,
                        'index_i': i,
                        'index_j': j
                    })

    def _sort_savings(self):
        """按节约量降序排序"""
        self.savings.sort(key=lambda x: x['saving'], reverse=True)

    def _merge_routes(self):
        """基于节约量合并路径，同时考虑约束条件"""
        # 为每条路径创建标识，用于快速查找
        route_map = {tuple(route['customers']): idx for idx, route in enumerate(self.routes)}

        for saving in self.savings:
            i_id = saving['i']
            j_id = saving['j']

            # 找到包含i和j的路径
            route_i = self._find_route_containing(i_id)
            route_j = self._find_route_containing(j_id)

            if not route_i or not route_j or route_i == route_j:
                continue  # 路径不存在或已在同一路径

            # 检查合并可行性
            if self._can_merge_routes(route_i, route_j, i_id, j_id):
                # 执行合并
                merged_route = self._merge_two_routes(route_i, route_j, i_id, j_id)

                # 更新路径列表
                self.routes.remove(route_i)
                self.routes.remove(route_j)
                self.routes.append(merged_route)

        # 分配车辆
        self._assign_vehicles_to_routes()

    def _can_merge_routes(self, route_i, route_j, i_id, j_id) -> bool:

        if route_i['single_vehicle'] or route_j['single_vehicle']:
            return False

        """检查两条路径是否可以合并"""
        # 1. 检查车辆容量约束
        total_weight = route_i['load_weight'] + route_j['load_weight']
        total_volume = route_i['load_volume'] + route_j['load_volume']

        height_restriction = route_i['height_restricted'] and route_j['height_restricted']

        # 找到最合适的车辆检查容量
        suitable_vehicle = self._find_suitable_vehicle(total_weight, total_volume, height_restriction)
        if not suitable_vehicle:
            return False

        # 2. 检查时间窗约束
        # 获取i在路径i中的位置和j在路径j中的位置
        i_pos = route_i['sequence'].index(i_id)
        j_pos = route_j['sequence'].index(j_id)

        # 检查路径i的终点是否是i，路径j的起点是否是j（适合合并的条件）
        if i_pos != len(route_i['sequence']) - 2 or j_pos != 1:
            return False

        # 计算合并后的时间约束
        warehouse = self._get_warehouse_location()
        last_departure_i = route_i['departure_times'][i_id]
        travel_time_ij = calculate_travel_time(
            calculate_distance(self.customer_map[i_id], self.customer_map[j_id]), self.customer_map[i_id], self.customer_map[j_id]
        )
        arrival_j = last_departure_i + travel_time_ij

        # 检查是否满足j的时间窗
        j_customer = self.customer_map[j_id]
        j_tw_start = parse_time(j_customer.time_window_start)
        j_tw_end = parse_time(j_customer.time_window_end)

        if arrival_j > j_tw_end:  # 到达时间晚于时间窗结束
            return False

        # 检查合并后返回仓库的时间是否在车辆可用时间内, 单司机工作时间不超过6小时
        vehicle_work_start_time = route_i['vehicle_work_start_time']
        service_time_j = get_service_time(j_customer)
        departure_j = max(arrival_j, j_tw_start) + service_time_j
        return_time = calculate_travel_time(
            calculate_distance(j_customer, warehouse), j_customer, warehouse
        )

        if departure_j + return_time > parse_time(suitable_vehicle.available_time_end):
            return False

        if departure_j + return_time - vehicle_work_start_time > 6 * 60:
            return False

        return True

    def _merge_two_routes(self, route_i, route_j, i_id, j_id) -> Dict:
        """合并两条路径"""
        # 创建新路径序列（移除重复的仓库点）
        new_sequence = route_i['sequence'][:-1] + route_j['sequence'][1:]

        # 计算新的距离和时间
        new_distance = route_i['total_distance'] + route_j['total_distance'] - \
                       calculate_distance(self.customer_map[i_id], self._get_warehouse_location()) - \
                       calculate_distance(self._get_warehouse_location(), self.customer_map[j_id]) + \
                       calculate_distance(self.customer_map[i_id], self.customer_map[j_id])

        # 合并装载信息
        new_weight = route_i['load_weight'] + route_j['load_weight']
        new_volume = route_i['load_volume'] + route_j['load_volume']

        # 合并客户列表
        new_customers = route_i['customers'] + route_j['customers']

        route_i['district'].update(route_j['district'])
        new_height_restricted = route_i['height_restricted'] or route_j['height_restricted']

        # 计算新的到达和离开时间
        new_arrival_times = {**route_i['arrival_times'], **route_j['arrival_times']}
        new_departure_times = {**route_i['departure_times'], **route_j['departure_times']}

        # 更新j的到达时间
        i_customer = self.customer_map[i_id]
        j_customer = self.customer_map[j_id]

        travel_time_ij = calculate_travel_time(
            calculate_distance(i_customer, j_customer), i_customer, j_customer
        )

        new_arrival_j = new_departure_times[i_id] + travel_time_ij
        new_departure_j = max(new_arrival_j, parse_time(j_customer.time_window_start)) + \
                          get_service_time(j_customer)

        new_arrival_times[j_id] = new_arrival_j
        new_departure_times[j_id] = new_departure_j

        return_time = calculate_travel_time(
            calculate_distance(j_customer, self._get_warehouse_location()), j_customer, self._get_warehouse_location()
        )

        return {
            'vehicle_id': None,
            'vehicle_type': None,
            'customers': new_customers,
            'sequence': new_sequence,
            'total_distance': new_distance,
            'total_time': new_departure_j + calculate_travel_time(
                calculate_distance(j_customer, self._get_warehouse_location()), j_customer, self._get_warehouse_location()
            ),
            'load_weight': new_weight,
            'load_volume': new_volume,
            # 多条不小于2个customer的如今合并时，需要更新后面的customer的时间数据
            'arrival_times': new_arrival_times,
            'departure_times': new_departure_times,
            'district': route_i['district'],
            'specified_vehicle': None,
            'height_restricted': new_height_restricted,
            'breakpoint': None,
            'single_vehicle': False,
            'vehicle_work_start_time': route_i['vehicle_work_start_time'],
            'vehicle_work_end_time': new_departure_j + return_time
        }

    def _assign_vehicles_to_routes(self):
        """为每条路径分配最合适的车辆"""
        selected_vehicles = set()
        for route in self.routes:
            # 找到能满足该路径需求的最合适车辆
            best_vehicle = None
            appropriate_vehicle = None
            for vehicle in self.problem.data_manager.vehicles:
                if vehicle.id in selected_vehicles:
                    continue
                if route['height_restricted'] and vehicle.vehicle_type.startswith("4.2"):
                    continue
                # todo 暂时只考虑体积约束
                # if vehicle.capacity_weight >= route['load_weight'] and vehicle.capacity_volume >= route['load_volume']:、
                if vehicle.capacity_volume >= route['load_volume']:
                    appropriate_vehicle = vehicle
                    # 满载率约束，暂时只考虑体积
                    if route['load_volume'] <= 0.85 * vehicle.capacity_volume:
                        best_vehicle = vehicle
                        break
            if best_vehicle or appropriate_vehicle:
                route['vehicle_id'] = best_vehicle.id if best_vehicle is not None else appropriate_vehicle.id
                route['vehicle_type'] = best_vehicle.vehicle_type if best_vehicle is not None else appropriate_vehicle.vehicle_type
                selected_vehicles.add(route['vehicle_id'])
                route['cost'] = self._calculate_transportation_cost(route)
                self.total_cost += route['cost']
            else:
                logger.warning(f"没有合适的车辆满足路径需求")

    # 辅助方法
    def _get_warehouse_location(self) -> Location:
        """获取仓库位置"""
        for loc in self.problem.data_manager.locations:
            if loc.location_type == 'warehouse':
                return loc
        raise ValueError("未找到仓库位置")

    def _find_route_containing(self, customer_id: str) -> Optional[Dict]:
        """找到包含指定客户的路径"""
        for route in self.routes:
            if customer_id in route['customers']:
                return route
        return None

    def _find_suitable_vehicle(self, weight: float, volume: float, height_restriction: bool) -> Optional[Vehicle]:
        """找到能满足重量和体积需求的车辆"""
        appropriate_vehicle = None
        for vehicle in self.problem.data_manager.vehicles:
            if height_restriction and vehicle.vehicle_type.startswith("4.2"):
                continue
            # if vehicle.capacity_weight >= weight and vehicle.capacity_volume >= volume:
            if vehicle.capacity_volume >= volume:
                appropriate_vehicle = vehicle
                # 满载率约束，暂时只考虑体积
                if volume <= 0.85 * vehicle.capacity_volume:
                    return vehicle
        return appropriate_vehicle

    def _calculate_customer_load(self, customer: Customer) -> Dict[str, float]:
        return calculate_customer_load(customer, self.product_map)

    def _calculate_transportation_cost(self, route:Dict[str, Any]):
        return calculate_transportation_cost(route, self.problem.data_manager.vehicle_costs)

    def _calculate_metrics(self):
        """计算解决方案的各项指标"""
        # 已在合并和分配车辆过程中计算


class VRPTWMain:
    """Main class orchestrating the entire VRPTW solving process"""

    def __init__(self):
        self.data_manager = DataManager()
        self.preprocessor = None
        self.problem = None
        self.output_manager = OutputManager()

    def run(self, algorithm: str = 'saving') -> bool:
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

            solver = SavingsAlgorithmSolver(self.problem, True)
            solution = solver.solve()

            logger.info(solution)

            # Step 5: Generate output
            logger.info("Generating output to csv_data/output/result.csv")
            self.output_manager.generate_output(solution, self.data_manager)

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
    # Run immediately for testing
    print("\n🔄 Running VRPTW solver with real vehicle data...")
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()