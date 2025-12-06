from method import runTrial
import graphing

def toList(a, add=0):
  return [x.item()+add for x in a]
def graphMethodResults(trials=5):
  for trial in range(trials):
    estimate, truth, stats = runTrial(40, scale=0.3, iters=50, SFrac=0.2, endDiscrim=10, Afrac=0.7)
    estNaive, _, _ = runTrial(40,naive=True, scale=0.3, iters=50, SFrac=0.2, endDiscrim=10, Afrac=0.7)
    graphing.plot_value_sets([toList(x, 1.5) for x in [
      truth-truth.mean(), estNaive[0], estimate[0],  estimate[1]
    ]], ["Ground truth", "Naive", "Estimate (smooth)", "Estimate (sharp)"],
    colors=["black","blue", "red","orange"], save_path="img/experiment1-"+str(trial))

def checkOrdinal(trials=10, N=40, scale=0.3, sfrac = 0.2):
  oursBad = 0
  naiveBad = 0
  naiveVeryBad = 0
  oursVeryBad = 0
  for trial in range(trials):
    estimate, truth, stats = runTrial(N, scale=scale, iters=50, SFrac=sfrac, endDiscrim=10, Afrac=0.7)
    estNaive, _, _ = runTrial(N,naive=True, scale=scale, iters=50, SFrac=sfrac, endDiscrim=10, Afrac=0.7)
    ours = toList(estimate[1])
    naive = toList(estNaive[0])
    flag1 = False
    flag2 = False
    for i in range(len(ours)-1):  
      if ours[i]>ours[i+1]:
        flag1 = True
      if naive[i]>naive[i+1]:
        flag2 = True
    if flag1:
      oursBad+=1
    if flag2:
      naiveBad+=1
    if max(naive) <= naive[2]:
      naiveVeryBad+=1
    if max(ours) <= ours[2]:
      oursVeryBad+=1
  print(f"{scale} & {oursBad} $ {oursVeryBad} & {naiveBad} & {naiveVeryBad}  \\\\")



#graphs in part 1
if True:
  graphMethodResults()

#scale table
if True:
  pass
  checkOrdinal(scale=0.15)
  checkOrdinal(scale=0.22)
  checkOrdinal(scale=0.29)
  checkOrdinal(scale=0.36)
  checkOrdinal(scale=0.42)

#weights over time graph
if True:
  estimate, truth, stats = runTrial(40,verbose=False, scale=0.25, iters=40, SFrac=0.2, Afrac=0.7, endDiscrim=10)
  print(estimate)
  graphing.plot_good_bad_trajectories(stats["weights"].clip(0,1),stats["schemers"], title="weights", fpath="img/experiment2--w")
  graphing.plot_good_bad_trajectories(stats["intrinsic"],stats["schemers"], title="intrinsic", fpath="img/experiment2--i")
  graphing.plot_good_bad_trajectories(stats["extrinsic"],stats["schemers"], title="extrinsic", fpath="img/experiment2--e")

  estimate, truth, stats = runTrial(40,verbose=False, scale=0.3, iters=50, SFrac=0.2, Afrac=0.7, endDiscrim=10)
  print(estimate)
  graphing.plot_good_bad_trajectories(stats["weights"].clip(0,1),stats["schemers"], title="weights", fpath="img/experiment3--w")
  graphing.plot_good_bad_trajectories(stats["intrinsic"],stats["schemers"], title="intrinsic", fpath="img/experiment3--i")
  graphing.plot_good_bad_trajectories(stats["extrinsic"],stats["schemers"], title="extrinsic", fpath="img/experiment3--e")