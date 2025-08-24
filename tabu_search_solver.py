import math
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import random
from copy import deepcopy

# 继承已有数据类
from main import Customer, Vehicle, Location, VRPTWSolver, VRPTWProblem, logger
from saving_algo_solver import SavingsAlgorithmSolver


class TabuSearchSolver(VRPTWSolver):
    """禁忌搜索算法求解VRPTW问题"""

    def __init__(self, problem: VRPTWProblem, tabu_size: int = 50, max_iter: int = 1000,
                 neighborhood_size: int = 50, aspiration_value: float = 0.1, enable_penalty=False, penalty_coeff={}):
        super().__init__(problem)
        self.tabu_list = []  # 禁忌表
        self.tabu_size = tabu_size  # 禁忌表大小
        self.max_iter = max_iter  # 最大迭代次数
        self.neighborhood_size = neighborhood_size  # 邻域大小
        self.aspiration_value = aspiration_value  # 愿望值（改进比例）

        self.best_solution = None
        self.best_cost = float('inf')
        self.best_score = float('inf')
        self.best_penalty = float('inf')
        self.current_solution = None
        self.current_cost = 0.0
        self.enable_penalty = enable_penalty
        self.penalty_coeff = penalty_coeff

    def solve(self) -> Dict[str, Any]:
        """实现禁忌搜索算法"""
        logger.info("使用禁忌搜索算法求解VRPTW问题...")

        try:
            # 1. 生成初始解
            self._generate_initial_solution()

            if not self.current_solution:
                raise ValueError("无法生成初始解")

            self.best_solution = deepcopy(self.current_solution)
            self.best_score = self._calculate_solution_score(self.current_solution)
            self.current_score = self.best_score

            # 2. 迭代搜索
            for iter in range(self.max_iter):
                # 生成邻域解
                neighborhood = self._generate_neighborhood()

                # 评估邻域解
                best_neighbor, best_neighbor_cost = self._evaluate_neighborhood(neighborhood)

                if not best_neighbor:
                    continue  # 没有找到可行解

                # 更新当前解
                self.current_solution = best_neighbor
                self.current_cost = best_neighbor_cost

                # 更新禁忌表
                self._update_tabu_list(best_neighbor)

                # 更新最优解
                if best_neighbor_cost < self.best_cost:
                    self.best_solution = deepcopy(best_neighbor)
                    self.best_cost = best_neighbor_cost
                    logger.info(f"迭代 {iter}: 找到更优解，成本 {self.best_cost:.2f}")

            solution = {
                'algorithm': 'Tabu Search',
                'routes': self.best_solution,
                'total_cost': self.best_cost,
                'vehicles_used': len(self.best_solution),
                'status': 'solved'
            }

            self.solution = solution
            return solution

        except Exception as e:
            logger.error(f"禁忌搜索算法求解出错: {e}")
            return {
                'algorithm': 'Tabu Search',
                'routes': [],
                'total_cost': 0,
                'status': 'failed',
                'error': str(e)
            }

    def _generate_initial_solution(self):
        """生成初始解（可以使用节约算法的结果作为初始解）"""
        # 使用节约算法生成初始解
        savings_solver = SavingsAlgorithmSolver(self.problem)
        savings_solution = savings_solver.solve()

        if savings_solution['status'] == 'solved':
            self.current_solution = savings_solution['routes']
            self.best_cost = savings_solution['total_cost']
        else:
            raise Exception("Saving algorithm fails. No available initial solution!")

    def _generate_neighborhood(self) -> List[List[Dict]]:
        """生成邻域解"""
        neighborhood = []

        for _ in range(self.neighborhood_size):
            # 随机选择一种操作生成邻域解
            operation = random.choice(['swap', 'insert', 'reverse', 'relocate'])

            if operation == 'swap':
                # 交换两个客户在不同路径中的位置
                neighbor = self._swap_customers(deepcopy(self.current_solution))
            elif operation == 'insert':
                # 将一个客户插入到另一条路径
                neighbor = self._insert_customer(deepcopy(self.current_solution))
            elif operation == 'reverse':
                # 反转路径中的部分客户顺序
                neighbor = self._reverse_segment(deepcopy(self.current_solution))
            else:  # relocate
                # 将一个客户从一条路径移动到另一条路径
                neighbor = self._relocate_customer(deepcopy(self.current_solution))

            if neighbor and self._is_solution_feasible(neighbor):
                neighborhood.append(neighbor)

        return neighborhood

    def _swap_customers(self, solution: List[Dict]) -> Optional[List[Dict]]:
        """交换两个客户在不同路径中的位置"""
        if len(solution) < 2:
            return None  # 至少需要两条路径

        # 随机选择两条不同的路径
        route_idx1, route_idx2 = random.sample(range(len(solution)), 2)
        route1 = solution[route_idx1]
        route2 = solution[route_idx2]

        if len(route1['customers']) < 1 or len(route2['customers']) < 1:
            return None  # 路径中至少需要有一个客户

        # 随机选择两个客户
        cust_idx1 = random.randint(0, len(route1['customers']) - 1)
        cust_idx2 = random.randint(0, len(route2['customers']) - 1)

        cust1_id = route1['customers'][cust_idx1]
        cust2_id = route2['customers'][cust_idx2]

        # 执行交换
        route1['customers'][cust_idx1] = cust2_id
        route2['customers'][cust_idx2] = cust1_id

        # 重新计算路径信息
        self._recompute_route(route1)
        self._recompute_route(route2)

        return solution

    def _insert_customer(self, solution: List[Dict]) -> Optional[List[Dict]]:
        """将一个客户插入到同一条路径的不同位置"""
        if len(solution) < 1:
            return None

        # 随机选择一条路径
        route_idx = random.randint(0, len(solution) - 1)
        route = solution[route_idx]

        if len(route['customers']) < 2:
            return None  # 路径中至少需要有两个客户

        # 随机选择一个客户和一个新位置
        cust_idx = random.randint(0, len(route['customers']) - 1)
        new_pos = random.randint(0, len(route['customers']) - 1)

        if cust_idx == new_pos:
            return None  # 位置相同，无需插入

        # 执行插入
        customer = route['customers'].pop(cust_idx)
        route['customers'].insert(new_pos, customer)

        # 重新计算路径信息
        self._recompute_route(route)

        return solution

    def _reverse_segment(self, solution: List[Dict]) -> Optional[List[Dict]]:
        """反转路径中的部分客户顺序"""
        if len(solution) < 1:
            return None

        # 随机选择一条路径
        route_idx = random.randint(0, len(solution) - 1)
        route = solution[route_idx]

        if len(route['customers']) < 2:
            return None  # 路径中至少需要有两个客户

        # 随机选择两个位置
        start_idx = random.randint(0, len(route['customers']) - 2)
        end_idx = random.randint(start_idx + 1, len(route['customers']) - 1)

        # 执行反转
        route['customers'][start_idx:end_idx + 1] = reversed(route['customers'][start_idx:end_idx + 1])

        # 重新计算路径信息
        self._recompute_route(route)

        return solution

    def _relocate_customer(self, solution: List[Dict]) -> Optional[List[Dict]]:
        """将一个客户从一条路径移动到另一条路径"""
        if len(solution) < 1:
            return None

        # 随机选择源路径和目标路径
        source_idx = random.randint(0, len(solution) - 1)
        target_idx = random.randint(0, len(solution) - 1)

        if source_idx == target_idx and len(solution[source_idx]['customers']) < 2:
            return None  # 同一路径至少需要有两个客户

        source_route = solution[source_idx]
        target_route = solution[target_idx]

        if len(source_route['customers']) < 1:
            return None  # 源路径至少需要有一个客户

        # 随机选择一个客户
        cust_idx = random.randint(0, len(source_route['customers']) - 1)
        customer = source_route['customers'].pop(cust_idx)

        # 随机选择插入位置
        insert_pos = random.randint(0, len(target_route['customers']))
        target_route['customers'].insert(insert_pos, customer)

        # 重新计算路径信息
        self._recompute_route(source_route)
        self._recompute_route(target_route)

        # 如果源路径为空，删除它
        if not source_route['customers']:
            solution.pop(source_idx)

        return solution

    def _recompute_route(self, route: Dict):
        """重新计算路径的距离、时间和装载信息"""
        warehouse = self._get_warehouse_location()

        # 重建序列
        route['sequence'] = [warehouse.id] + route['customers'] + [warehouse.id]

        # 重新计算距离
        total_distance = 0.0
        for i in range(len(route['sequence']) - 1):
            loc1 = self._get_location_by_id(route['sequence'][i])
            loc2 = self._get_location_by_id(route['sequence'][i + 1])
            total_distance += self._calculate_distance(loc1, loc2)

        route['total_distance'] = total_distance

        # 重新计算装载信息
        total_weight = 0.0
        total_volume = 0.0
        for cust_id in route['customers']:
            customer = self._get_customer_by_id(cust_id)
            load = self._calculate_customer_load(customer)
            total_weight += load['weight']
            total_volume += load['volume']

        route['load_weight'] = total_weight
        route['load_volume'] = total_volume

        # 重新计算时间信息
        arrival_times = {}
        departure_times = {}
        current_time = self._parse_time(self._get_vehicle_by_id(route['vehicle']).available_time_start)
        current_loc = warehouse

        for cust_id in route['customers']:
            customer = self._get_customer_by_id(cust_id)

            # 计算到达时间
            distance = self._calculate_distance(current_loc, customer)
            travel_time = self._calculate_travel_time(distance)
            arrival_time = current_time + travel_time

            # 考虑时间窗
            tw_start = self._parse_time(customer.time_window_start)
            tw_end = self._parse_time(customer.time_window_end)
            effective_arrival = max(arrival_time, tw_start)

            if effective_arrival > tw_end:
                # 违反时间窗约束，标记为不可行
                route['feasible'] = False
                return

            # 计算离开时间
            service_time = self._get_service_time(customer)
            departure_time = effective_arrival + service_time

            # 更新
            arrival_times[cust_id] = effective_arrival
            departure_times[cust_id] = departure_time
            current_time = departure_time
            current_loc = customer

        # 返回仓库的时间
        distance = self._calculate_distance(current_loc, warehouse)
        current_time += self._calculate_travel_time(distance)

        route['arrival_times'] = arrival_times
        route['departure_times'] = departure_times
        route['total_time'] = current_time
        route['feasible'] = True

    def _evaluate_neighborhood(self, neighborhood: List[List[Dict]]) -> Tuple[Optional[List[Dict]], float]:
        """评估邻域解，选择最佳解"""
        if not neighborhood:
            return None, float('inf')

        # 计算每个邻域解的成本
        evaluated = []
        for solution in neighborhood:
            if self._is_solution_feasible(solution):
                cost = self._calculate_solution_cost(solution)
                evaluated.append((solution, cost))

        if not evaluated:
            return None, float('inf')

        # 按成本排序
        evaluated.sort(key=lambda x: x[1])

        # 检查禁忌表和愿望准则
        for solution, cost in evaluated:
            solution_hash = self._hash_solution(solution)

            # 检查是否在禁忌表中
            if solution_hash in self.tabu_list:
                # 检查愿望准则：如果解比当前最优解好很多，则接受
                if cost < self.best_cost * (1 - self.aspiration_value):
                    return solution, cost
                continue
            else:
                # 非禁忌解，直接接受
                return solution, cost

        # 如果所有好的解都被禁忌，选择成本最低的禁忌解
        return evaluated[0]

    def _update_tabu_list(self, solution: List[Dict]):
        """更新禁忌表"""
        solution_hash = self._hash_solution(solution)

        # 添加新解到禁忌表
        self.tabu_list.append(solution_hash)

        # 如果禁忌表超过最大大小，移除最早的条目
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)

    def _is_solution_feasible(self, solution: List[Dict]) -> bool:
        """检查解是否可行（满足所有约束）"""
        for route in solution:
            # 检查可行性标记
            if not route.get('feasible', True):
                return False

            # 检查车辆容量约束
            vehicle = self._get_vehicle_by_id(route['vehicle'])
            if (route['load_weight'] > vehicle.capacity_weight or
                    route['load_volume'] > vehicle.capacity_volume):
                return False

            # 检查时间约束
            if route['total_time'] > self._parse_time(vehicle.available_time_end):
                return False

            # 检查每个客户的时间窗约束
            for cust_id, arrival_time in route['arrival_times'].items():
                customer = self._get_customer_by_id(cust_id)
                tw_end = self._parse_time(customer.time_window_end)
                if arrival_time > tw_end:
                    return False

        return True

    def _calculate_solution_score(self, solution: List[Dict], cost: float) -> float:

        for route in solution:
            vehicle = self._get_vehicle_by_id(route['vehicle'])
            # 成本 = 距离 * 单位距离成本
            total_cost += route['total_distance'] * vehicle.cost_per_km

        return total_cost

    def _hash_solution(self, solution: List[Dict]) -> str:
        """生成解的哈希值，用于禁忌表"""
        # 简单哈希：按路径和客户顺序生成字符串
        route_strs = []
        for route in solution:
            route_str = "-".join(route['customers'])
            route_strs.append(route_str)

        return "|".join(sorted(route_strs))  # 排序确保顺序不影响哈希

    # 辅助方法（与节约算法中的类似）
    def _get_warehouse_location(self) -> Location:
        for loc in self.problem.data_manager.locations:
            if loc.location_type == 'warehouse':
                return loc
        raise ValueError("未找到仓库位置")

    def _get_customer_by_id(self, customer_id: str) -> Customer:
        for customer in self.problem.data_manager.customers:
            if customer.id == customer_id:
                return customer
        raise ValueError(f"未找到ID为{customer_id}的客户")

    def _get_vehicle_by_id(self, vehicle_id: str) -> Vehicle:
        for vehicle in self.problem.data_manager.vehicles:
            if vehicle.id == vehicle_id:
                return vehicle
        raise ValueError(f"未找到ID为{vehicle_id}的车辆")

    def _get_location_by_id(self, location_id: str):
        """通过ID获取位置（仓库或客户）"""
        if location_id == 'warehouse':
            return self._get_warehouse_location()
        for customer in self.problem.data_manager.customers:
            if customer.id == location_id:
                return customer
        raise ValueError(f"未找到ID为{location_id}的位置")

    def _calculate_customer_load(self, customer: Customer) -> Dict[str, float]:
        total_weight = 0.0
        total_volume = 0.0

        for product_id, quantity in customer.demand.items():
            product = next((p for p in self.problem.data_manager.products if p.id == product_id), None)
            if product:
                total_weight += product.weight_per_unit * quantity
                total_volume += product.volume_per_unit * quantity

        return {'weight': total_weight, 'volume': total_volume}

    def _calculate_distance(self, loc1, loc2) -> float:
        return math.hypot(loc1.longitude - loc2.longitude, loc1.latitude - loc2.latitude) * 100

    def _calculate_travel_time(self, distance: float) -> float:
        avg_speed = 40.0  # 平均速度，公里/小时
        return (distance / avg_speed) * 60  # 转换为分钟

    def _get_service_time(self, customer: Customer) -> float:
        return 10.0  # 默认10分钟

    def _parse_time(self, time_str: str) -> float:
        try:
            time_obj = datetime.strptime(time_str, "%H:%M:%S")
            return time_obj.hour * 60 + time_obj.minute
        except ValueError:
            try:
                time_obj = datetime.strptime(time_str, "%H:%M")
                return time_obj.hour * 60 + time_obj.minute
            except ValueError:
                return 0.0