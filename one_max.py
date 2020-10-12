import numpy as np
import mlrose_hiive

fitness = mlrose_hiive.OneMax()
problem = mlrose_hiive.DiscreteOpt(length=200, fitness_fn=fitness, maximize=True, max_val=2)
SEED=42
output_directory = "./one_max_testing_1"

#####################
# RHC
#####################
# rhc = mlrose_hiive.RHCRunner(problem=problem,
#                        experiment_name="RHC_one_max",
#                        output_directory=output_directory,
#                        seed=SEED,
#                        iteration_list=[4096],
#                        max_attempts=500, #5000
#                        restart_list=[25,75])  # [25,75,100]
# rhc_run_stats, rhc_run_curves = rhc.run()

#####################
# Simulated Annealing
#####################
sa = mlrose_hiive.SARunner(problem=problem,
                     experiment_name="SA_one_max",
                     output_directory=output_directory,
                     seed=SEED,
                     iteration_list=[8192],
                     max_attempts=500,
                     temperature_list=[1,10,50,100,250,500,1000],
                     decay_list=[mlrose_hiive.ExpDecay,mlrose_hiive.GeomDecay])
sa_run_stats, sa_run_curves = sa.run()


#####################
# Genetic Alg
#####################
# ga = mlrose_hiive.GARunner(problem=problem,
#                      experiment_name="GA_one_max",
#                      output_directory=output_directory,
#                      seed=SEED,
#                      iteration_list=2**np.arange(10),
#                      max_attempts=500,
#                      population_sizes=[150,200,300],
#                      mutation_rates=[0.1, 0.2,0.4])
# ga_run_stats, ga_run_curves = ga.run()
# #
# #
# # #####################
# # # MIMIC
# # #####################
# mimic = mlrose_hiive.MIMICRunner(problem=problem,
#                            experiment_name="MIMIC_one_max",
#                            output_directory=output_directory,
#                            seed=SEED,
#                            iteration_list=2**np.arange(10),
#                            population_sizes=[150,200,300],
#                            max_attempts=500,
#                            keep_percent_list=[0.25,0.5,0.75],
#                            use_fast_mimic=True)
# mimic_run_stats, mimic_run_curves = mimic.run()