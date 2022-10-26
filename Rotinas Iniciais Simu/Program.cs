using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Models;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.TensorExtensionMethods;
using TorchSharp.Modules;

using System.Collections;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

namespace RotinasIniciais
{
  class Program
  {
    static void Main(string[] args)
    {
      int memory_size = Convert.ToInt32(5 * Math.Pow(10, 5));
      int batch_size = 32;
      double gamma = 0.99;
      decimal exploration_max = 1.0M;
      decimal exploration_min = 0.01M;
      decimal exploration_decay = 0.995M;
      float learning_rate = 0.05f;
      double tau = 0.125;
      int n_actions = 2;
      int n_inputs = 2;
      int layers = 2;
      int neurons = 24;
      int episodes = 10_000;

      //var dqn_keras = new dqn_keras(memory_size, batch_size, gamma, exploration_max, exploration_min, exploration_decay, learning_rate, tau, n_actions, n_inputs);
      //dqn_keras.run(episodes);

      var dqn_torch = new dqn_torch(memory_size, batch_size, gamma, exploration_max, exploration_min, exploration_decay, learning_rate, tau, n_actions, n_inputs, layers, neurons);
      dqn_torch.run(episodes);
    }
  }
}