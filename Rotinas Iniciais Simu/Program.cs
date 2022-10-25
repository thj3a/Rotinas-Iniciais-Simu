using System.Collections.Generic;
using System.Diagnostics;
using Tensorflow.NumPy;
using Tensorflow;
using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.TensorExtensionMethods;
using System.Collections;
using System.Linq;
using TorchSharp.Modules;
using System.Runtime.InteropServices;
using Tensorflow.Keras.Layers;

namespace RotinasIniciais
{
  class Program
  {
    static void Main(string[] args)
    {
      int memory_size = Convert.ToInt32(5 * Math.Pow(10, 5));
      int batch_size = 32;
      double gamma = 0.99;
      double exploration_max = 1.0;
      double exploration_min = 0.01;
      double exploration_decay = 0.995;
      double learning_rate = 0.001;
      double tau = 0.125;
      int n_actions = 2;
      int n_inputs = 2;
      int episodes = 1000;

      //var dqn_keras = new dqn_keras(memory_size, batch_size, gamma, exploration_max, exploration_min, exploration_decay, learning_rate, tau, n_actions, n_inputs);
      //dqn_keras.run(episodes);

      var dqn_torch = new dqn_torch(memory_size, batch_size, gamma, exploration_max, exploration_min, exploration_decay, learning_rate, tau, n_actions, n_inputs);
      dqn_torch.run(episodes);
    }
  }
}