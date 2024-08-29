# MLP - Neuronal Network

## Enuntul Temei
Antrenati o retea neuronala MLP cu cel putin un strat ascuns pentru identificarea cifrei dintr-o imagine, cu ajutorul pytorch.

Aveti setul de date de la MNIST ce contine cateva zeci de mii de astfel de exemple de imagini (input-ul), impreuna cu cifra (output-ul dorit).

## Explicații la Tema Rezolvata

### 1. Introducere
Cu ajutorul PyTorch si NumPy am realizat 3 retele neuronale care identifica cifra dintr-o imagine. Am realizat 3 deoarece am vrut sa observ diferentele dintre utilizarea unei librarii specifice crearii de retele neuronale si cele facute manual cu si fara optimizer. Fiecare retea neuronala are atasata si un fisier `.txt` in care sunt salvate: `learning rate`, `epocile`, `batch size`, si `acuratetea`. Toate retelele neuronale au 2 straturi ascunse cu **392** si **100** de neuroni, functiile de activare `ReLU` si `Softmax`, pentru ultimul strat, si functia de cost `Cross-Entropy Loss`.

### 2. Structura Proiectului
Explică structura proiectului tău. De exemplu:
- **Fisierul `first_neuronal_network.py`**: Retea neuronala construita cu ajutorul framework-ului PyTorch. Am observat diferente majore intre utilizare cu bias sau fara, respectiv optimizer SGD simplu si cu momentum.
  1. *optimizer* cu *momentum = 0.90*, *learning rate = 0.01* : acuratete de **97.81%**
  2. *optimezer* cu *momentum = 0.90* impreuna cu *dampening = 25%*, *learning rate = 0.01*: acuratete de **97.74%**
  3. *optimizer simplu* si *bias = False*, *learning rate = 0.01*` :  acuratete de **95.86%**
  4. *optimizer simplu* si *bias = True*, *learning rate = 0.01* :  acuratete de **86.20%**
  5. *optimizer* si *momentum = 0.99*, *learning rate = 0.01* : acuratete de **10.30**
- **Fisierul `neuronal_network.py`**: Retea neuronala construita manual doar cu ajutorul librariei NumPy, *learning rate = 0.001*, *epoch = 35*, *batch size = 25* si s-a ajuns la o acuratete de **77.82%**.
- **Fisierul `nn-optimizer.py`**: Retea neuronala construita manual doar cu ajutorul librariei NumPy dar cu optimizerului **Stochastic Gradient Descent**, *learning rate = 0.001*, *epoch = 35*, *batch size = 25* si s-a ajuns la o acuratete de **81.40%**.

### 3. Metodologie
Descrie pașii pe care i-ai urmat pentru a rezolva tema. De exemplu:
1. **Colectarea Datelor**: Am colectat datele din setul de date de la MNIST.
2. **Preprocesarea Datelor**: Datele sunt impartite in train_set, valid_set si test_set.
3. **Implementarea Algoritmului**: Folosirea a 2 a straturi ascunse cu **392** si **100** de neuroni, functiile de activare `ReLU` si `Softmax`, pentru ultimul strat, si functia de cost `Cross-Entropy Loss`.


### 4.1. Rezultate 
S-au putut observa beneficiile utilizarii framework-ului PyTorch, optimizat pentru performanta, utilizarea eficienta a optimizerului si cum implementarea manuala si actualizarea vectorilor de weights si bias contribuie negativ la performanta retelei neuronale. In plus, folosirea optimizerului **`SGD`** pentru estimarea gradinetului la fiecare pas aduce o imbunatatire mare, insa folosirea impreuna cu **`momentum`**, accelerand in directia gradientului si reducand oscilarea, face ca reteaua neuronala sa invete si mai bine.



