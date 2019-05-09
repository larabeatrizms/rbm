from rbm import RBM
import numpy as np

rbm = RBM(num_visible = 6, num_hidden= 3)

base = np.array([[0,1,1,1,0,1],
                 [1,1,0,1,1,1],
                 [0,1,0,1,0,1],
                 [0,1,1,1,0,1], 
                 [1,1,0,1,0,1],
                 [1,1,0,1,1,1]])


filmes = ["Freddy x Jason", "O ultimato Bourne", "Star Trek",
         "Exterminador do Futuro", "Norbit", "Star Wars"]

rbm.train(base, max_epochs=5000)
#rbm.weights

usuario1 = np.array([[0,1,0,1,0,0]])

camada_escondida = rbm.run_visible(usuario1)
recomendacao = rbm.run_hidden(camada_escondida)

for i in range(len(usuario1[0])):
    #print(usuario1[0,i])
    if usuario1[0,i] == 0 and recomendacao[0,i] == 1:
        print(filmes[i])
    
