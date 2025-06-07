import main

model = main.WidrowHoff.load('Models/model_Widrow-Hoff_lr0.005_ep20000.npz', main.X, main.T)

main.evaluate_model(model, main.X, main.T, main.letters_list)