import logging
import math
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime
import random
import traceback
from copy import deepcopy

# 继承已有数据类
from main import Customer, Vehicle, Location, VRPTWSolver, VRPTWProblem, logger, DataManager, OutputManager, \
    DataPreprocessor
from saving_algo_solver import SavingsAlgorithmSolver, calculate_distance, calculate_travel_time, get_service_time, \
    parse_time, calculate_transportation_cost, calculate_customer_load, extract_district

ACROSS_DISTRICTS = "acrosss_districts"
OVER_LOADING_85 = "over_loading_85"
OVER_LOADING_90 = "over_loading_90"


class TabuSearchSolver(VRPTWSolver):
    """禁忌搜索算法求解VRPTW问题"""

    def __init__(self, problem: VRPTWProblem, tabu_size: int = 50, max_iter: int = 1000,
                 neighborhood_size: int = 50, aspiration_value: float = 0.1, enable_penalty=True, penalty_coeff={}):
        super().__init__(problem)
        self.tabu_list = []  # 禁忌表
        self.tabu_size = tabu_size  # 禁忌表大小
        self.max_iter = max_iter  # 最大迭代次数
        self.neighborhood_size = neighborhood_size  # 邻域大小
        self.aspiration_value = aspiration_value  # 愿望值（改进比例）

        self.saving_solver = SavingsAlgorithmSolver(self.problem, True)

        self.best_solution = None
        self.best_cost = float('inf')
        self.best_score = float('inf')
        self.best_penalty = float('inf')
        self.current_solution = None
        self.enable_penalty = enable_penalty
        self.penalty_coeff = penalty_coeff
        self.print_details = True

    def solve(self) -> Dict[str, Any]:
        """实现禁忌搜索算法"""
        logger.info("使用寻优算法求解VRPTW问题...")

        try:
            # 1. 生成初始解
            self._generate_initial_solution()
            self._establish_info_map()

            if not self.current_solution:
                raise ValueError("无法生成初始解")

            warehouse = self._get_warehouse_location()
            excluded_routes = []
            excluded_routes_indices = []
            for cus_id, customer in self.customer_map.items():
                tw_end = self._parse_time(customer.time_window_end)
                distance = calculate_distance(warehouse, customer)
                # travel_time = self._calculate_travel_time(distance)
                travel_time = calculate_travel_time(distance, warehouse, customer)
                if travel_time > tw_end or travel_time > 60*3:
                    selected_route, index = self._find_route_containing(self.current_solution, cus_id)
                    excluded_routes.append(selected_route)
                    excluded_routes_indices.append(index)
                    logger.info(f"剔除过远线路-客户{customer.name}：距离{distance}km, 用时{travel_time}min")
            explored_routes = [self.current_solution[i] for i in range(len(self.current_solution)) if i not in excluded_routes_indices]
            fixed_cost = sum([route['cost'] for route in excluded_routes])
            logger.info(f"{len(excluded_routes_indices)}条路径由于需要剔除距离过远的customer，因此被固定。剩余待搜索的路径数：{len(explored_routes)}")

            self.best_solution = deepcopy(explored_routes)
            self.current_solution = deepcopy(explored_routes)
            self.best_cost = sum([route['cost'] for route in explored_routes])
            self.best_penalty = self._calculate_solution_penalty(explored_routes)
            self.best_score = self.best_cost + self.best_penalty
            self.current_score = self.best_score
            logger.info(f"Initial solution，cost： {self.best_cost:.2f}, penalty：{self.best_penalty:.2f}")


            # 2. 迭代搜索
            for iter in range(self.max_iter):
                # 生成邻域解
                neighborhood = self._generate_neighborhood()

                # 评估邻域解
                best_neighbor, best_neighbor_cost, best_neighbor_penalty = self._evaluate_neighborhood(neighborhood)

                if not best_neighbor:
                    if self.print_details:
                        logger.info(f"迭代 {iter}: 未找到更优解，成本 {self.best_cost:.2f}, 惩罚 {best_neighbor_penalty:.2f}")
                    continue  # 没有找到可行解

                # 更新当前解
                self.current_solution = best_neighbor

                # 更新禁忌表
                self._update_tabu_list(best_neighbor)

                # 更新最优解
                if not self.enable_penalty:
                    if best_neighbor_cost < self.best_cost:
                        self.best_solution = deepcopy(best_neighbor)
                        self.best_cost = best_neighbor_cost
                        logger.info(f"*迭代 {iter}: 找到更优解，成本 {self.best_cost:.2f}")
                    elif self.print_details:
                        logger.info(f"迭代 {iter}: 未找到更优解，成本 {self.best_cost:.2f}")
                else:
                    if best_neighbor_cost + best_neighbor_penalty < self.best_score:
                        self.best_solution = deepcopy(best_neighbor)
                        self.best_score = best_neighbor_cost + best_neighbor_penalty
                        logger.info(f"*迭代 {iter}: 找到更优解，成本： {best_neighbor_cost:.2f},"
                                    f" 惩罚值：{best_neighbor_penalty:.2f}")
                    elif self.print_details:
                        logger.info(f"迭代 {iter}: 未找到更优解，成本： {best_neighbor_cost:.2f},"
                                    f" 惩罚值：{best_neighbor_penalty:.2f}")

            self.best_solution = self.best_solution + excluded_routes
            self._calculate_interval_distance_and_time(self.best_solution)
            solution = {
                'algorithm': 'Tabu Search',
                'routes': self.best_solution,
                'total_cost': self.best_cost + fixed_cost,
                'vehicles_used': len(self.best_solution),
                'status': 'solved'
            }

            self.solution = solution
            return solution

        except Exception as e:
            logger.error(f"禁忌搜索算法求解出错: {e}")
            tb_str = traceback.format_exc()
            print(tb_str)  # 打印字符串形式的堆栈信息
            return {
                'algorithm': 'Tabu Search',
                'routes': [],
                'total_cost': 0,
                'status': 'failed',
                'error': str(e)
            }

    def _find_route_containing(self, routes, customer_id: str):
        """找到包含指定客户的路径"""
        for i in range(len(routes)):
            if customer_id in routes[i]['customers']:
                return routes[i], i
        return None

    def _establish_info_map(self):
        self.vehicle_map = self.saving_solver.vehicle_map
        self.customer_map = self.saving_solver.customer_map
        self.product_map = self.saving_solver.product_map
        # for vehicle in self.problem.data_manager.vehicles:
        #     self.vehicle_map[vehicle.id] = vehicle
        # for customer in self.problem.data_manager.customers:
        #     self.customer_map[customer.id] = customer
        # for product in self.problem.data_manager.products:
        #     self.product_map[product.id] = product

    def _calculate_interval_distance_and_time_per_route(self, route):
        route['interval_distance'] = {}
        route['interval_time'] = {}
        vehicle_type = "4.2" if route['vehicle_type'] == "4.2m厢式货车" else ""
        for i in range(len(route['customers'])):
            cus_id = route['customers'][i]
            customer = self.customer_map[cus_id]
            if i == 0:
                distance = calculate_distance(self._get_warehouse_location(), customer, vehicle_type)
                route['interval_time'][cus_id] = calculate_travel_time(distance, self._get_warehouse_location(), customer, vehicle_type)
            else:
                prev_customer = self.customer_map[route['customers'][i-1]]
                distance = calculate_distance(prev_customer, customer, vehicle_type)
                route['interval_time'][cus_id] = calculate_travel_time(distance, prev_customer, customer, vehicle_type)
            route['interval_distance'][cus_id] = distance

    def _calculate_interval_distance_and_time(self, routes):
        for route in routes:
            self._calculate_interval_distance_and_time_per_route(route)

    def _generate_initial_solution(self):
        """生成初始解（可以使用节约算法的结果作为初始解）"""
        # 使用节约算法生成初始解
        savings_solution = self.saving_solver.solve()

        if savings_solution['status'] == 'solved':
            self.current_solution = savings_solution['routes']
            self.best_cost = savings_solution['total_cost']
        else:
            raise Exception("Saving algorithm fails. No available initial solution!")

    def _calculate_solution_penalty_value(self, routes):
        penalty_value = {
            ACROSS_DISTRICTS: 0,
            OVER_LOADING_85: 0,
            OVER_LOADING_90: 0
        }
        for route in routes:
            if len(route['district']) >= 4:
                penalty_value[ACROSS_DISTRICTS] += 1
            vehicle_volume_capacity = self.vehicle_map[route['vehicle_id']].capacity_volume
            load_ratio = route['load_volume'] / vehicle_volume_capacity
            if 0.85 < load_ratio <= 0.9:
                penalty_value[OVER_LOADING_85] += 1
            elif load_ratio > 0.9:
                penalty_value[OVER_LOADING_90] += 1

        return penalty_value

    def _calculate_solution_penalty(self, routes):
        penalty_value = self._calculate_solution_penalty_value(routes)
        penalty = 0
        route_total_num = len(routes)
        for key, value in penalty_value.items():
            if key == ACROSS_DISTRICTS:
                across_ratio = float(value) / route_total_num
                if across_ratio > 0.3:
                    penalty += self.penalty_coeff.get(key, 1) * (across_ratio - 0.3)
            else:
                penalty += self.penalty_coeff.get(key, 1) * value

        return penalty

    def _calculate_total_transportation_cost(self, routes):
        return sum([route['cost'] for route in routes])

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
        if route1['single_vehicle'] or route2['single_vehicle']:
            return None

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

        route1['modified'] = True
        route2['modified'] = True

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
        route['modified'] = True

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
        route['modified'] = True

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

        if source_route['single_vehicle'] or target_route['single_vehicle']:
            return None

        if len(source_route['customers']) < 1:
            solution.pop(source_idx)
            return None  # 源路径至少需要有一个客户

        # 随机选择一个客户
        cust_idx = random.randint(0, len(source_route['customers']) - 1)
        customer = source_route['customers'].pop(cust_idx)

        # 随机选择插入位置
        insert_pos = random.randint(0, len(target_route['customers']))
        target_route['customers'].insert(insert_pos, customer)

        # 重新计算路径信息
        if len(source_route['customers']) >= 1:
            self._recompute_route(source_route)
        self._recompute_route(target_route)

        # 如果源路径为空，删除它
        source_route['modified'] = True
        target_route['modified'] = True
        if not source_route['customers'] or len(source_route['customers']) < 1:
            solution.pop(source_idx)


        return solution

    def _recompute_route(self, route: Dict):
        """重新计算路径的距离、时间和装载信息"""
        warehouse = self._get_warehouse_location()

        # 重建序列
        route['sequence'] = [warehouse.id] + route['customers'] + [warehouse.id]

        total_distance = 0

        # 重新计算装载信息
        total_weight = 0.0
        total_volume = 0.0
        for cust_id in route['customers']:
            customer = self.customer_map[cust_id]
            load = self._calculate_customer_load(customer)
            total_weight += load['weight']
            total_volume += load['volume']

        route['load_weight'] = total_weight
        route['load_volume'] = total_volume

        route['height_restricted'] = True in [self.customer_map[i].height_restricted for i in route['customers']]
        suitable_vehicle = self.saving_solver._find_suitable_vehicle(total_weight, total_volume,
                                                                     route['height_restricted'])
        suitable_vehicle_type = "4.2" if suitable_vehicle.vehicle_type == "4.2m厢式货车" else ""
        if not suitable_vehicle:
            route['feasible'] = False
            return

        # 重新计算时间信息
        arrival_times = {}
        departure_times = {}
        district = set()
        customer_num = len(route['customers'])

        current_loc = self.customer_map[route['customers'][0]]
        service_time = get_service_time(current_loc)

        first_distance = calculate_distance(self._get_warehouse_location(), self.customer_map[route['customers'][0]], suitable_vehicle_type)
        total_distance += first_distance
        first_travel_time = calculate_travel_time(first_distance, self._get_warehouse_location(), self.customer_map[route['customers'][0]],
                                                  suitable_vehicle_type)
        tw_start = parse_time(current_loc.time_window_start)
        arrival_times[route['customers'][0]] = tw_start
        work_start_time = tw_start - first_travel_time
        current_time = tw_start + service_time
        departure_times[route['customers'][0]] = current_time

        for cust_id in route['customers']:
            customer = self.customer_map[cust_id]
            if customer.delivery_type == "单点配送" and customer_num >= 2:
                route['feasible'] = False
                return

            # 计算到达时间
            distance = calculate_distance(current_loc, customer, suitable_vehicle_type)
            total_distance += distance
            travel_time = calculate_travel_time(distance, current_loc, customer, suitable_vehicle_type)
            arrival_time = current_time + travel_time

            # 考虑时间窗
            tw_start = self._parse_time(customer.time_window_start)
            tw_end = self._parse_time(customer.time_window_end)
            effective_arrival = max(arrival_time, tw_start)

            if effective_arrival > tw_end:
                # 违反时间窗约束，标记为不可行
                route['feasible'] = False
                return

            district.add(extract_district(customer.address))

            # 计算离开时间
            service_time = get_service_time(customer)
            departure_time = effective_arrival + service_time

            # 更新
            arrival_times[cust_id] = effective_arrival
            departure_times[cust_id] = departure_time
            current_time = departure_time
            current_loc = customer

        vehicle_work_end_time = current_time

        if vehicle_work_end_time - work_start_time > 60 * 6:
            route['feasible'] = False
            return

        route['arrival_times'] = arrival_times
        route['departure_times'] = departure_times
        route['total_time'] = vehicle_work_end_time - work_start_time
        route['total_distance'] = total_distance
        route['district'] = district
        route['vehicle_work_start_time'] = work_start_time
        route['vehicle_work_end_time'] = vehicle_work_end_time
        route['feasible'] = True
        route['vehicle_id'] = None
        route['vehicle_type'] = None

    def _evaluate_neighborhood(self, neighborhood: List[List[Dict]]) -> Union[
        tuple[None, float, float], tuple[list[dict], int, int]]:
        """评估邻域解，选择最佳解"""
        if not neighborhood:
            return None, float('inf'), float('inf')

        # 计算每个邻域解的成本
        evaluated = []
        for solution in neighborhood:
            is_feasible = self._assign_vehicles_and_validate_feasibility(solution)
            if is_feasible:
                cost = self._calculate_total_transportation_cost(solution)
                penalty = self._calculate_solution_penalty(solution)
                evaluated.append((solution, [cost, penalty]))

        if not evaluated:
            return None, float('inf'), float('inf')

        # todo 可以设置不同的排序方式
        evaluated.sort(key=lambda x: (x[1][0], x[1][1]))

        # 检查禁忌表和愿望准则
        for solution, value in evaluated:
            solution_hash = self._hash_solution(solution)

            # 检查是否在禁忌表中
            if solution_hash in self.tabu_list:
                # 检查愿望准则：如果解比当前最优解好很多，则接受
                if value[0] < self.best_cost * (1 - self.aspiration_value):
                    return solution, value[0], value[1]
                continue
            else:
                # 非禁忌解，直接接受
                return solution, value[0], value[1]

        # 如果所有好的解都被禁忌，选择成本最低的禁忌解
        return evaluated[0][0], evaluated[0][1][0], evaluated[0][1][1]

    def _assign_vehicles_to_routes(self, routes):
        """为每条路径分配最合适的车辆"""
        selected_vehicles = set()
        for route in routes:
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
                route[
                    'vehicle_type'] = best_vehicle.vehicle_type if best_vehicle is not None else appropriate_vehicle.vehicle_type
                selected_vehicles.add(route['vehicle_id'])
                route['cost'] = self._calculate_transportation_cost(route)
            else:
                logger.warning(f"没有合适的车辆满足路径需求")

    def _update_tabu_list(self, solution: List[Dict]):
        """更新禁忌表"""
        solution_hash = self._hash_solution(solution)

        # 添加新解到禁忌表
        self.tabu_list.append(solution_hash)

        # 如果禁忌表超过最大大小，移除最早的条目
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)

    def _is_solution_feasible(self, solution: List[Dict]) -> bool:
        """检查解是否可行 粗略检查"""
        for route in solution:
            # 检查可行性标记
            if not route.get('feasible', True):
                return False

            # 检查每个客户的时间窗约束
            # for cust_id, arrival_time in route['arrival_times'].items():
                # customer = self.customer_map[cust_id]
                # tw_end = self._parse_time(customer.time_window_end)
                # if arrival_time > tw_end:
                #     return False

        return True

    def _assign_vehicles_and_validate_feasibility(self, solution: List[Dict]) -> bool:
        """检查解是否可行（满足所有约束）"""
        for route in solution:
            # 检查可行性标记
            if not route.get('feasible', True):
                return False
        # 分配车辆
        self._assign_vehicles_to_routes(solution)
        for route in solution:
            if route['vehicle_id'] is None:
                logging.info(
                    f"vehicle is is None")
                return False

            vehicle = self.vehicle_map[route['vehicle_id']]
            # if (route['load_weight'] > vehicle.capacity_weight or
            #         route['load_volume'] > vehicle.capacity_volume):
            if route['load_volume'] > vehicle.capacity_volume:
                logger.info(f"vehicle: {route['vehicle_id']}, route_load is {route['load_volume']}, capacity is {vehicle.capacity_volume}")
                return False

            # # 检查时间约束
            if route['vehicle_work_end_time'] - route['vehicle_work_start_time'] > 60*6:
                return False
            # if route['vehicle_work_end_time'] > self._parse_time(vehicle.available_time_end):
            #     logger.info(f"vehicle: {route['vehicle_id']}, route_load is {route['load_volume']}, capacity is {vehicle.capacity_volume}")
            #     return False

            # 检查每个客户的时间窗约束
            # for cust_id, arrival_time in route['arrival_times'].items():
            #     customer = self.customer_map[cust_id]
            #     tw_end = self._parse_time(customer.time_window_end)
            #     if arrival_time > tw_end:
            #         return False

        return True

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

    def _get_location_by_id(self, location_id: str):
        """通过ID获取位置（仓库或客户）"""
        if location_id == 'warehouse':
            return self._get_warehouse_location()
        return self.customer_map[location_id]

    def _calculate_customer_load(self, customer: Customer) -> Dict[str, float]:
        return calculate_customer_load(customer, self.product_map)

    # def _calculate_distance(self, loc1, loc2) -> float:
    #     return calculate_distance(loc1, loc2)
    #
    # def _calculate_travel_time(self, distance: float) -> float:
    #     return calculate_travel_time(distance)
    #
    # def _calculate_travel_time(self, distance, loc1, loc2) -> float:
    #     return calculate_travel_time(distance, loc1, loc2)

    def _get_service_time(self, customer: Customer) -> float:
        return get_service_time(customer)

    def _parse_time(self, time_str: str) -> float:
        return parse_time(time_str)

    def _calculate_transportation_cost(self, route: Dict[str, Any]):
        return calculate_transportation_cost(route, self.problem.data_manager.vehicle_costs)


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

            penalty_coeff = {
                ACROSS_DISTRICTS: 10,
                OVER_LOADING_85: 5,
                OVER_LOADING_90: 10
            }
            solver = TabuSearchSolver(self.problem, enable_penalty=True, penalty_coeff=penalty_coeff)
            solution = solver.solve()

            # logger.info(solution)
            logger.info(f"Total cost is {solution['total_cost']:.2f}, vehicle_required_num is {solution['vehicles_used']}")

            # Step 5: Generate output
            logger.info("Generating output to csv_data/output/result.csv")
            self.output_manager.generate_output(solution, self.data_manager)

            logger.info("VRPTW solving pipeline completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Error in VRPTW pipeline: {e}")
            return False


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
