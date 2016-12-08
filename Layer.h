#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

class Layer {
private:
  int numNeurons;
  std::vector<Neuron*>* neurons;

public:
  Layer() {}

  // Create the vector of neurons for this layer.
  Layer(int neuronCount, int inputsPerNeuron) {
    numNeurons = neuronCount;
    neurons = new std::vector<Neuron*>(numNeurons);

    for (int i = 0; i < neuronCount; i++) {
      (*neurons)[i] = new Neuron(inputsPerNeuron);
    }
  }

  ~Layer() {
    for (int i = 0; i < neurons->size(); i++) {
      delete (*neurons)[i];
    }
    delete neurons;
  }

  // Get the number of neurons in this layer.
  int neuronCount() const {
    return numNeurons;
  }

  // Get the neuron at the given position.
  Neuron *getNeuron(int n) const {
    return (*neurons)[n];
  }

  void printNeurons() {
    for (int i = 0; i < neurons->size(); i++) {
      std::cout << "Neuron #" << i << "\n";
      (*neurons)[i]->printWeights();
    }
    std::cout << "\n\n";
  }
};
