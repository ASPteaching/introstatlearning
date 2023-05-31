library(tfruns)

# simple case
# training_run('MNIST_flags.r',
#              flags = c(hl1 = 200, hl2 = 100))
# loop case
system.time(
  for (hl1 in c(200, 300))
    for (hl2 in c(50, 150))
      training_run('MNIST_flags.r', 
                   flags = c(hl1 = hl1))
)

# Show last completed run
latest_run()
# simple case, show all runs
ls_runs()
# Show all runs with improved presentation
View(ls_runs())
# show selection items from all runs
ls_runs(metric_val_accuracy > 0.94, 
        order = metric_val_accuracy)
# compare_runs() visual comparison of two training runs. 
compare_runs() # Default is comare two last runs

