import numpy as np
import mlrose_hiive

fitness = mlrose_hiive.FourPeaks(t_pct=0.1)
problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
SEED=42
output_directory = "./four_peaks_testing_4"

#####################
# RHC
#####################
rhc = mlrose_hiive.RHCRunner(problem=problem,
                       experiment_name="RHC_four_peaks",
                       output_directory=output_directory,
                       seed=SEED,
                       iteration_list=2**np.arange(10),
                       max_attempts=5000, #5000
                       restart_list=[25,75,100])
rhc_run_stats, rhc_run_curves = rhc.run()


#####################
# Simulated Annealing
#####################
sa = mlrose_hiive.SARunner(problem=problem,
                     experiment_name="SA_four_peaks",
                     output_directory=output_directory,
                     seed=SEED,
                     iteration_list=2**np.arange(13),
                     max_attempts=5000, #5000
                     temperature_list=[1,10,50,100,250,500,1000],
                     decay_list=[mlrose_hiive.ExpDecay, mlrose_hiive.GeomDecay])
sa_run_stats, sa_run_curves = sa.run()


# #####################
# # Genetic Alg
# #####################
ga = mlrose_hiive.GARunner(problem=problem,
                     experiment_name="GA_four_peaks",
                     output_directory=output_directory,
                     seed=SEED,
                     iteration_list=2**np.arange(10),
                     max_attempts=500,
                     population_sizes=[150,200,300],
                     mutation_rates=[0.1,0.4,0.5,0.6])
ga_run_stats, ga_run_curves = ga.run()


# #####################
# # MIMIC
# #####################
mimic = mlrose_hiive.MIMICRunner(problem=problem,
                           experiment_name="MIMIC_four_peaks",
                           output_directory=output_directory,
                           seed=SEED,
                           iteration_list=2**np.arange(13),
                           population_sizes=[150,200,300],
                           max_attempts=500,
                           keep_percent_list=[0.25, 0.5, 0.75],
                           use_fast_mimic=True)
mimic_run_stats, mimic_run_curves = mimic.run()