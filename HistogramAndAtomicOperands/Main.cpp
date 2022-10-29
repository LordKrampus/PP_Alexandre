#include <iostream>
#include "Methods.h"

using namespace std;

#define HIST_SIZE 11

/*
* Retorna um valor arredondado da nota correspondendo a um indíce no histograma de notas
*/
int DetectRange(const float nota) {
	int range;
	float fracao;

	fracao = nota - (int)nota;
	range = (int)nota;
	if (fracao >= 0.5f)
		range += 1;

	return range;
}

/*
* calcula os valores no histograma de notas
*/
void CalcHist(const float* notas, int size, int* hist) {
	for (int i = 0; i < size; i++) {
		hist[DetectRange(notas[i])]++;
	}
}

void ExecuteBySincHost(const float* notas, const int size, int* hist) {
	CalcHist(notas, size, hist);
}

void RestartHist(int* hist) {
	for (int i = 0; i < HIST_SIZE; i++)
		hist[i] = 0;
}

void PrintHist(const int* hist) {
	cout << "\"Grade Hist\":" << endl;

	int count;
	for (int i = 0; i < HIST_SIZE; i++) {
		cout << i << "\t|";

		count = hist[i];
		for (int e = 0; e < count; e++)
			cout << '#';

		cout << endl;
	}
}

int main() {
	float notas[] = { 1.2, 2.2, 1.5, 6.0, 7.9, 9.1, 9.5, 0.1, 0.0, 0.5, 0.6, 0.9 };

	cout << "Notas:" << endl;
	for (int i = 0; i < sizeof(notas)/sizeof(float); i++)
		cout << notas[i] << " ";

	int* hist;
	hist = (int*)malloc(HIST_SIZE * sizeof(int));

	cout << "\n\n(Host)" << endl;
	RestartHist(hist);
	CalcHist(notas, sizeof(notas)/sizeof(float), hist);
	PrintHist(hist);

	cout << "\n(Device)" << endl;
	RestartHist(hist);
	CuCalcHist(notas, sizeof(notas), hist, HIST_SIZE * sizeof(int));
	PrintHist(hist);
	//!!! como as treads que acessam o mesmo indice no histograma o fazem simultâneamente, 
	// as operações se sobrepões na localidade de memória - o valor anterior é somando em 1
	// para todos os casos em que o valor anterior se trata do mesmo - resultando no mesmo.

	cout << "\n(Device + Atomic Operands)" << endl;
	RestartHist(hist);
	CuAtomicCalcHist(notas, sizeof(notas), hist, HIST_SIZE * sizeof(int));
	PrintHist(hist);

	return 0;
}