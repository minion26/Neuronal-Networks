# Single Layer Perceptron

## Enuntul Temei
Antrenati 10 perceptroni pentru identificarea cifrei dintr-o imagine.

In esenta, vom avea cate un perceptron pentru fiecare cifra, care va invata sa diferentieze intre cifra respectiva si restul cifrelor. Prin asamblarea acestora putem deduce ce cifra ar fi intr-o imagine specifica.

Aveti setul de date de la MNIST ce contine cateva zeci de mii de astfel de exemple de imagini (input-ul), impreuna cu cifra (output-ul dorit al ansamblului).

## Explicații la Tema Rezolvata

### 1. Introducere
Am utilizat atat Perceptronul Simplu, cat si ADALINE. 

### 2. Structura Proiectului
Explică structura proiectului tău. De exemplu:
- **Fisierul `single-perceptron-layer.py`**: Conține o clasa Perceptron in care am utilizat algoritmul Perceptronului Simplu.
- **Fisierul `adaline-single-perceptron-layer.py`**: Adaptarea codului pentru a folosi caracteristicile perceptronului ADALINE.

### 3. Metodologie
Descrie pașii pe care i-ai urmat pentru a rezolva tema. De exemplu:
1. **Colectarea Datelor**: Am colectat datele din setul de date de la MNIST.
2. **Preprocesarea Datelor**: Datele sunt impartite in train_set, valid_set si test_set.
3. **Implementarea Algoritmului**: Am implementat o clasa Perceptron in care am initializat rata de invatare, epocile, numarul de intrari si numarul de perceptroni (10 pentru fiecare cifra de la 0 la 9). Aceasta clasa are o functie *prediction* in care se foloseste o functie step sau liniara, *evaluate* in  care se calculeaza acuratetea pentru fiecare epoca, respectiv *train* unde se antreneaza modelul, se calculeaza eroarea in functie de tipul de perceptron folosit, simplu sau ADALINE, si se recalculeaza bias-ul si vectorul de weights.

### 4.1. Rezultate - Simple Perceptron
| Epoch | Validation Accuracy |
|-------|---------------------|
| 1     | 0.8196              |
| 2     | 0.8283              |
| 3     | 0.8282              |
| 4     | 0.8184              |
| 5     | 0.8104              |
| 6     | 0.834               |
| 7     | 0.8117              |
| 8     | 0.8281              |
| 9     | 0.8364              |
| 10    | 0.7995              |

![image](https://github.com/user-attachments/assets/da23a087-9abb-4b4f-aa72-77003e15d541)


### 4.2. Rezultate - ADALINE Perceptron
| Epoch | Validation Accuracy |
|-------|---------------------|
| 1     | 0.7368              |
| 2     | 0.7370              |
| 3     | 0.7379              |
| 4     | 0.7378              |
| 5     | 0.7380              |
| 6     | 0.7378              |
| 7     | 0.7375              |
| 8     | 0.7378              |
| 9     | 0.7378              |
| 10    | 0.7381              |

![image](https://github.com/user-attachments/assets/8139132c-29da-49e8-9e8d-5e41c7645013)



