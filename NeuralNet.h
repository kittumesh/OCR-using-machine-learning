#include "Neuron.h"
#include "Layer.h"

class NeuralNet {
private:
  int numInputs;
  int numOutputs;
  int numHiddenLayers;
  int numNeuronsPerLayer;
  double learningRate;
  double responseThreshold;

  std::vector<Layer*>* layers;
  double* outputLayer;

public:
  NeuralNet(int inputs,
            int outputs,
            int hiddenLayers,
            int neuronsPerLayer,
            double alpha,
            double threshold);

  ~NeuralNet();

  double* getWeights() const;

  // Compute the outputs from a given set of inputs.
  void feedForward(std::vector<double>* inputs,
                   std::vector<double>* outputLayer,
                   const double bias);

  // Back propagate the errors to update the weights.
  void backPropagate(std::vector<double>* outputs, int teacher);

  // Sigmoid response function.
  inline double sigmoid(double activation);

  // Derivative of sigmoid response function.
  inline double sigmoidPrime(double activation);

  void print() {
    for (int i = 1; i < layers->size(); i++) {
      std::cout << "Layerr #" << i << "\n";
      (*layers)[i]->printNeurons();
    }
  }
};
