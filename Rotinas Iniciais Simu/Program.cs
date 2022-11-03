//using Tensorflow;
//using Tensorflow.Keras;
//using Tensorflow.Keras.ArgsDefinition;
//using Tensorflow.Keras.Engine;
//using Tensorflow.Keras.Layers;
//using Tensorflow.Keras.Models;
//using Tensorflow.NumPy;
//using static Tensorflow.Binding;
//using static Tensorflow.KerasApi;

using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.TensorExtensionMethods;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torch.Tensor;
using static TorchSharp.TensorExtensionMethods;

using System.Collections;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using Keras.Models;

namespace RotinasIniciais
{
  class Program
  {
    static void Main(string[] args)
    {
      int memory_size = Convert.ToInt32(5 * Math.Pow(10, 5));
      int batch_size = 32;
      float gamma = 0.99f;
      float exploration_max = 1.0f;
      float exploration_min = 0.01f;
      float exploration_decay = 0.995f;
      float learning_rate = 0.005f;
      float tau = 0.125f;
      int n_outputs = 2;
      int n_inputs = 2;
      int layers = 2;
      int neurons = 24;
      int episodes = 1_000;
      var device = torch.cuda.is_available() ? torch.device("CUDA") : torch.device("cpu");

      var model = new dqn_torch(batch_size,
                                    gamma,
                                    exploration_max,
                                    exploration_min,
                                    exploration_decay,
                                    learning_rate,
                                    tau,
                                    n_outputs,
                                    n_inputs,
                                    layers,
                                    neurons,
                                    "model",
                                    device);
      var aux_model = new dqn_torch(batch_size,
                                    gamma,
                                    exploration_max,
                                    exploration_min,
                                    exploration_decay,
                                    learning_rate,
                                    tau,
                                    n_outputs,
                                    n_inputs,
                                    layers,
                                    neurons,
                                    "aux_model",
                                    device);

      Adam optimizer = new(model.parameters(), learning_rate);
      
      simple_env env = new simple_env();
      replay_memory memory = new replay_memory(memory_size);
      
      List<int> rewards = new();
      List<float> losses = new();
      (Tensor state, bool done) = env.reset();
      Tensor next_state = state.clone();
      int reward = 0;
      int action = -1;

      for (int ep = 0; ep < episodes; ep++)
      {
        (state, done) = env.reset();
        List<int> r = new();
        List<float> l = new();
        while (!done)
        {
          action = model.get_action(state.clone());
          (next_state, reward, done) = env.step(state.clone(), action);
          memory.remember(state.clone(), action, reward, next_state.clone(), done);
          //if (ep % 32 == 0)
          l.Add(replay(memory, model, aux_model, optimizer));
          state = next_state.clone();
          r.Add(reward);
        }
        rewards.Add(r.Sum());
        losses.Add(torch.mean(torch.tensor(l)).item<float>());
        //Console.Write("Rewards in ep.: ");
        //Console.WriteLine();
        //Console.Write("Losses in ep.:");
        //l.ForEach(x => Console.Write($"{x.ToString()}..."));
        // Console.WriteLine($"Next state: [{String.Join(",", state)}], Reward: {reward}, Done?: {done}");
        //Console.WriteLine();
        Console.WriteLine($" --- Episode {ep}, Ep. Reward: {rewards[ep]}, Ep. Loss: {losses[ep]}, Exploration: {model.exploration_max}");
      }

    }
    public static float replay(replay_memory memory, dqn_torch model, Module<Tensor, Tensor> aux_model, optim.Optimizer optimizer)
    {

      //for (int i=0; i < this.batch_size; i++)
      //{
      //  samples.Add(memory[rnd.Next(memory_size)]);
      //}

      // -=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=- APPROACH 1
      //foreach ((var state, var action, var reward, var next_state, var done) in samples)
      //{
      //  var y_hat = model.forward(state);
      //  var y = model.forward(state);
      //  if (done)
      //  {
      //    y[0][action] = reward;
      //  }
      //  else
      //  {
      //    var Q_next = model.forward(next_state);
      //    var Q_next_max = torch.max(Q_next);

      //    y[0][action] = torch.tensor(reward) + (this.gamma * Q_next_max);
      //    // Q_next.print(); Q_next_max.print();
      //  }
      //  var criterion = torch.nn.MSELoss(Reduction.Sum);
      //  var output = criterion.forward(y_hat, y).to(device);
      //  model.zero_grad();
      //  output.backward();
      //  optimizer.step();
      //  //output.print(); y.print(); y_hat.print();
      //  loss.Add(output.item<float>());
      //}
      //return torch.mean(torch.tensor(loss.ToArray())).item<float>();

      // -=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=- APPROACH 2

      if (memory.size < model.batch_size) return -1f;
      List<(Tensor state, int action, float reward, Tensor next_state, bool done)> samples = memory.sample();
      var loss = new List<float>();
      var _states = new float[32, 2];
      var _targets = new float[32, 2];

      var _states_ = torch.cat(samples.Select(x => x.state).ToList());
      
      for (int i = 0, max = samples.Count(); i < max; i++)
      {
        _states[i, 0] = samples[i].state[0][0].item<float>();
        _states[i, 1] = samples[i].state[0][1].item<float>();
      }
      var states = torch.tensor(_states);
      Tensor targets;

        targets = aux_model.forward(states);

        for (int i = 0, imax = samples.Count(); i < imax; i++)
        {
          if (samples[i].done)
          {
            targets[i][samples[i].action] = samples[i].reward;
          }
          else
          {
            var next_state_q_values = aux_model.forward(samples[i].next_state);
            var next_state_q_value = torch.max(next_state_q_values, 1).values.item<float>();
            targets[i][samples[i].action] = samples[i].reward + (model.gamma * next_state_q_value);
          }
        }
      model.train();
      using (var d = torch.NewDisposeScope())
      {
        optimizer.zero_grad();
        var prediction = model.forward(states);
        var output = mse_loss(prediction, targets);
        output.print();
        output.backward();
        optimizer.step();
        return output.item<float>();
      }
    }

  }
}