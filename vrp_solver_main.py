from main import DataManager, OutputManager, DataPreprocessor, VRPTWProblem, logger
from saving_algo_solver import SavingsAlgorithmSolver
from tabu_search_solver import TabuSearchSolver


class VRPTWMain:
    """Main class orchestrating the entire VRPTW solving process"""

    def __init__(self):
        self.data_manager = DataManager()
        self.preprocessor = None
        self.problem = None
        self.solvers = {
            'saving': SavingsAlgorithmSolver,
            'tabu_search': TabuSearchSolver
        }
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

            # Step 4: Solve the problem
            if algorithm not in self.solvers:
                logger.error(f"Unknown algorithm: {algorithm}")
                return False

            solver_class = self.solvers[algorithm]
            solver = solver_class(self.problem)
            solution = solver.solve()

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

if __name__ == "__main__":

    # Run immediately for testing
    print("\n🔄 Running VRPTW solver with real vehicle data...")
    try:
        print("VRPTW Solver - Vehicle Routing Problem with Time Windows")
        print("=" * 60)

        # Initialize main solver
        vrptw_main = VRPTWMain()

        # Run with default greedy algorithm
        success = vrptw_main.run(algorithm='saving')

        if success:
            print("\n✅ VRPTW solving completed successfully!")
            print("Check csv_data/output/result.csv for results")
        else:
            print("\n❌ VRPTW solving failed. Check logs for details.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()