import json
import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from datetime import datetime
import random
from copy import deepcopy

# 继承已有数据类
from main import Customer, Vehicle, Product, Location, VRPTWSolver, VRPTWProblem, logger, DataManager, OutputManager, \
    DataPreprocessor


class SavingsAlgorithmSolver(VRPTWSolver):
    """带时间窗的节约算法求解VRPTW问题"""

    def __init__(self, problem: VRPTWProblem):
        super().__init__(problem)
        self.savings = []  # 存储节约量
        self.routes = []  # 存储路径
        self.total_cost = 0.0

    def solve(self) -> Dict[str, Any]:
        """实现带时间窗的节约算法"""
        logger.info("使用带时间窗的节约算法求解VRPTW问题...")

        try:
            # 1. 初始化：为每个客户创建单独路径
            self._initialize_routes()

            # 2. 计算所有客户对之间的节约量
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
            return {
                'algorithm': 'Savings with Time Windows',
                'routes': [],
                'total_cost': 0,
                'status': 'failed',
                'error': str(e)
            }

    def _initialize_routes(self):
        """初始化路径：每个客户单独一条路径（仓库-客户-仓库）"""
        warehouse = self._get_warehouse_location()

        for customer in self.problem.data_manager.customers:
            # 计算客户需求
            load = self._calculate_customer_load(customer)

            # 计算往返距离
            to_customer = self._calculate_distance(warehouse, customer)
            return_distance = self._calculate_distance(customer, warehouse)

            # 计算时间
            travel_time = self._calculate_travel_time(to_customer)
            service_time = self._get_service_time(customer)
            return_time = self._calculate_travel_time(return_distance)

            # 时间窗处理
            tw_start = self._parse_time(customer.time_window_start)
            tw_end = self._parse_time(customer.time_window_end)
            # todo 加上这辆车的available_start_time
            arrival_time = max(travel_time, tw_start)
            departure_time = arrival_time + service_time

            # todo 加上运输费用计算
            route = {
                'vehicle_id': None,  # 尚未分配车辆
                'customers': [customer.id],
                'sequence': [warehouse.id, customer.id, warehouse.id],
                'total_distance': to_customer + return_distance,
                'total_time': travel_time + service_time + return_time,
                'load_weight': load['weight'],
                'load_volume': load['volume'],
                'arrival_times': {customer.id: arrival_time},
                'departure_times': {customer.id: departure_time}
            }

            self.routes.append(route)

    def _calculate_savings(self):
        """计算所有客户对之间的节约量"""
        warehouse = self._get_warehouse_location()

        for i, customer_i in enumerate(self.problem.data_manager.customers):
            for j, customer_j in enumerate(self.problem.data_manager.customers):
                if i >= j:
                    continue  # 避免重复计算

                # 计算节约量: s_ij = c(0,i) + c(j,0) - c(i,j)
                # todo 这里计算方式不对，需要计算的是运输成本，注意方向
                c0i = self._calculate_distance(warehouse, customer_i)
                cj0 = self._calculate_distance(customer_j, warehouse)
                cij = self._calculate_distance(customer_i, customer_j)

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
        """检查两条路径是否可以合并"""
        # 1. 检查车辆容量约束
        total_weight = route_i['load_weight'] + route_j['load_weight']
        total_volume = route_i['load_volume'] + route_j['load_volume']

        # 找到最合适的车辆检查容量
        suitable_vehicle = self._find_suitable_vehicle(total_weight, total_volume)
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
        travel_time_ij = self._calculate_travel_time(
            self._calculate_distance(self._get_customer_by_id(i_id), self._get_customer_by_id(j_id))
        )
        arrival_j = last_departure_i + travel_time_ij

        # 检查是否满足j的时间窗
        j_customer = self._get_customer_by_id(j_id)
        j_tw_start = self._parse_time(j_customer.time_window_start)
        j_tw_end = self._parse_time(j_customer.time_window_end)

        # todo check是否卸完货的时间需要不迟于窗结束的时候
        if arrival_j > j_tw_end:  # 到达时间晚于时间窗结束
            return False

        # 检查合并后返回仓库的时间是否在车辆可用时间内
        service_time_j = self._get_service_time(j_customer)
        departure_j = max(arrival_j, j_tw_start) + service_time_j
        return_time = self._calculate_travel_time(
            self._calculate_distance(j_customer, warehouse)
        )

        if departure_j + return_time > self._parse_time(suitable_vehicle.available_time_end):
            return False

        return True

    def _merge_two_routes(self, route_i, route_j, i_id, j_id) -> Dict:
        """合并两条路径"""
        # 创建新路径序列（移除重复的仓库点）
        new_sequence = route_i['sequence'][:-1] + route_j['sequence'][1:]

        # 计算新的距离和时间
        new_distance = route_i['total_distance'] + route_j['total_distance'] - \
                       self._calculate_distance(self._get_customer_by_id(i_id), self._get_warehouse_location()) - \
                       self._calculate_distance(self._get_warehouse_location(), self._get_customer_by_id(j_id)) + \
                       self._calculate_distance(self._get_customer_by_id(i_id), self._get_customer_by_id(j_id))

        # 合并装载信息
        new_weight = route_i['load_weight'] + route_j['load_weight']
        new_volume = route_i['load_volume'] + route_j['load_volume']

        # 合并客户列表
        new_customers = route_i['customers'] + route_j['customers']

        # 计算新的到达和离开时间
        new_arrival_times = {**route_i['arrival_times'], **route_j['arrival_times']}
        new_departure_times = {**route_i['departure_times'], **route_j['departure_times']}

        # 更新j的到达时间
        i_customer = self._get_customer_by_id(i_id)
        j_customer = self._get_customer_by_id(j_id)

        travel_time_ij = self._calculate_travel_time(
            self._calculate_distance(i_customer, j_customer)
        )

        new_arrival_j = new_departure_times[i_id] + travel_time_ij
        new_departure_j = max(new_arrival_j, self._parse_time(j_customer.time_window_start)) + \
                          self._get_service_time(j_customer)

        new_arrival_times[j_id] = new_arrival_j
        new_departure_times[j_id] = new_departure_j

        return {
            'vehicle_id': None,
            'customers': new_customers,
            'sequence': new_sequence,
            'total_distance': new_distance,
            'total_time': new_departure_j + self._calculate_travel_time(
                self._calculate_distance(j_customer, self._get_warehouse_location())
            ),
            'load_weight': new_weight,
            'load_volume': new_volume,
            # 多条不小于2个customer的如今合并时，需要更新后面的customer的时间数据
            'arrival_times': new_arrival_times,
            'departure_times': new_departure_times
        }

    def _assign_vehicles_to_routes(self):
        """为每条路径分配最合适的车辆"""
        for route in self.routes:
            # 找到能满足该路径需求的最合适车辆
            best_vehicle = None
            min_cost = float('inf')

            for vehicle in self.problem.data_manager.vehicles:
                if (vehicle.capacity_weight >= route['load_weight'] and
                        vehicle.capacity_volume >= route['load_volume']):
                    # 计算使用该车辆的成本 todo 考虑最大装载率
                    cost = route['total_distance'] * vehicle.cost_per_km
                    if cost < min_cost:
                        min_cost = cost
                        best_vehicle = vehicle

            if best_vehicle:
                route['vehicle_id'] = best_vehicle.id
                self.total_cost += min_cost
            else:
                logger.warning(f"没有合适的车辆满足路径需求")

    # 辅助方法
    def _get_warehouse_location(self) -> Location:
        """获取仓库位置"""
        for loc in self.problem.data_manager.locations:
            if loc.location_type == 'warehouse':
                return loc
        raise ValueError("未找到仓库位置")

    def _get_customer_by_id(self, customer_id: str) -> Customer:
        """通过ID获取客户"""
        for customer in self.problem.data_manager.customers:
            if customer.id == customer_id:
                return customer
        raise ValueError(f"未找到ID为{customer_id}的客户")

    def _find_route_containing(self, customer_id: str) -> Optional[Dict]:
        """找到包含指定客户的路径"""
        for route in self.routes:
            if customer_id in route['customers']:
                return route
        return None

    def _find_suitable_vehicle(self, weight: float, volume: float) -> Optional[Vehicle]:
        """找到能满足重量和体积需求的车辆"""
        for vehicle in self.problem.data_manager.vehicles:
            # todo 考虑装载率约束
            if vehicle.capacity_weight >= weight and vehicle.capacity_volume >= volume:
                return vehicle
        return None

    def _calculate_customer_load(self, customer: Customer) -> Dict[str, float]:
        """计算客的总重量和总体积需求"""
        total_weight = 0.0
        total_volume = 0.0

        for product_id, quantity in customer.demand.items():
            product = next((p for p in self.problem.data_manager.products if p.id == product_id), None)
            if product:
                total_weight += product.weight_per_unit * quantity
                total_volume += product.volume_per_unit * quantity

        return {'weight': total_weight, 'volume': total_volume}

    def _calculate_distance(self, loc1, loc2) -> float:
        """计算两个位置之间的距离（使用已有方法或实现）"""
        # 这里可以使用实际的距离计算方法
        # 示例使用欧氏距离
        return math.hypot(loc1.longitude - loc2.longitude, loc1.latitude - loc2.latitude) * 100  # 简单转换为公里

    def _calculate_travel_time(self, distance: float) -> float:
        """根据距离计算旅行时间（分钟）"""
        avg_speed = 40.0  # 平均速度，公里/小时
        return (distance / avg_speed) * 60  # 转换为分钟

    def _get_service_time(self, customer: Customer) -> float:
        """获取客户的服务时间（分钟）"""
        # todo 需要根据客户需求数量动态计算
        return 10.0  # 默认10分钟

    def _parse_time(self, time_str: str) -> float:
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

            solver = SavingsAlgorithmSolver(self.problem)
            solution = solver.solve()

            logger.info(json.dumps(solution))

            # # Step 5: Generate output
            # logger.info("Generating output to csv_data/output/result.csv")
            # self.output_manager.generate_output(solution, self.data_manager)

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