using DequeNet;
using System;
using System.Collections.Generic;
using Microsoft.Collections;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torch.Tensor;
using static TorchSharp.TensorExtensionMethods;
using TorchSharp.Modules;
using System.ComponentModel.DataAnnotations;
using System.Runtime.CompilerServices;
using Microsoft.Collections.Extensions;

namespace RotinasIniciais
{
  public class dqn_torch : Module<Tensor, Tensor>
  {
    public int batch_size;
    public float gamma;
    public float exploration_max;
    public float exploration_min;
    public float exploration_decay;
    public float learning_rate;
    public float tau;
    public int n_actions;
    public int n_inputs;
    public int layers;
    public int neurons;

    public Random rnd = new Random();
    public Module<Tensor, Tensor> lin1 = Linear(2, 24);
    public Module<Tensor, Tensor> lin2 = Linear(24, 24);
    public Module<Tensor, Tensor> lin3 = Linear(24, 2);
    public Module<Tensor, Tensor> relu1 = ReLU();
    public Module<Tensor, Tensor> relu2 = ReLU();
    public Module<Tensor, Tensor> relu3 = ReLU();

    public dqn_torch(int batch_size,
                     float gamma,
                     float exploration_max,
                     float exploration_min,
                     float exploration_decay,
                     float learning_rate,
                     float tau,
                     int n_outputs,
                     int n_inputs,
                     int layers,
                     int neurons,
                     string name, 
                     torch.Device device = null) : base(name)
    {
      RegisterComponents();

      if (device != null && device.type == DeviceType.CUDA)
        this.to(device);

      this.batch_size = batch_size;
      this.gamma = gamma;
      this.exploration_max = exploration_max;
      this.exploration_min = exploration_min;
      this.exploration_decay = exploration_decay;
      this.learning_rate = learning_rate;
      this.tau = tau;
      this.n_inputs = n_inputs;
      this.n_actions = n_outputs;
      this.neurons = neurons;
      this.layers = layers;
    }
    
    public int get_action(Tensor state, bool should_explore = true)
    {
      int action = -1;
      if (should_explore)
      {
        exploration_max *= exploration_decay;
        exploration_max = Math.Max(exploration_min, exploration_max);
        if (this.rnd.NextDouble() < exploration_max)
          return rnd.Next(2);
      }

      var q_values = this.forward(state)[0];
      var best_action = torch.argmax(q_values);

      action = (int)best_action.item<long>();
      return action;
    }

    public override Tensor forward(Tensor input)
    {
      var l1y1 = lin1.forward(input);
      var l1y2 = relu1.forward(l1y1);
      var l2y1 = lin2.forward(l1y2);
      var l2y2 = relu2.forward(l2y1);
      var y = lin3.forward(l2y2);

      return y;
    }
  }

  public class simple_env
  {
    public int calc_reward(Tensor state, Tensor next_state, bool done)
    {
      var _state = new List<float>();
      var _next_state = new List<float>();
      for (long i = 0, imax = state.size()[0] + 1; i < imax; i++)
      {
        _state.Add(state[0][i].item<float>());
      }
      for (long i = 0, imax = next_state.size()[0]; i < imax; i++)
      {
        _next_state.Add(next_state[0][i].item<float>());
      }

      if (_next_state[0] == 0)
      {
        return -10;
      }
      else if (_next_state[0] == 3)
      {
        return 10;
      }
      else if (_state[1] == 1 && _next_state[0] == 1)
      {
        return 3;
      }
      else
      {
        return 0;
      }
    }

    public (Tensor, bool) reset()
    {
      var done = false;
      Tensor state = torch.tensor(new float[,] { { 2, 1 } });

      return (state, done);

    }

    public (Tensor, int, bool) step(Tensor state, int action)
    {
      Tensor next_state;
      var _state = new List<float>();
      var _next_state = new List<float>();
      var done = false;

      for (long i = 0, imax = state.size()[0] + 1; i < imax; i++)
      {
        _state.Add(state[0][i].item<float>());
        _next_state.Add((float)state[0][i].item<float>());
      }

      if (action == 0)
      {
        _next_state[0] -= 1;
        //Console.WriteLine("andei pra esquerda");
      }
      else
      {
        _next_state[0] += 1;
        //Console.WriteLine("andei pra direita");
      }

      if (_next_state[0] == 0 || _next_state[0] == 3)
      {
        done = true;
        //Console.WriteLine("caiu no fogo ou ganhou");
      }
      if (_next_state[1] == 1 && _next_state[0] == 1)
      {
        _next_state[1] = 0;
        //Console.WriteLine("diamante parou de existir");
      }
      next_state = torch.tensor(new float[,] { { _next_state[0], _next_state[1] } });
      int reward = calc_reward(state, next_state, done);

      //state.print();
      //next_state.print();
      //Console.WriteLine($"action: {action}, reward: {reward}");
      //Console.WriteLine("-=--=--=--=--=--=--=--=-");

      return (next_state, reward, done);
    }

  }

  public class replay_memory
  {
    public List<(Tensor state, int action, float reward, Tensor next_state, bool done)> memory;
    public Random rnd;
    public replay_memory(int capacity = 2000)
    {
      // For now, capacity is not working... Improve this later.
      memory = new();
      rnd = new();
    }
    public void remember(Tensor state, int action, float reward, Tensor next_state, bool done)
    {
      this.memory.Add((state, action, reward, next_state, done));
    }

    public List<(Tensor, int, float, Tensor, bool)> sample(int batch_size = 32)
    {
      List<(Tensor state, int action, float reward, Tensor next_state, bool done)> selected = this.memory.Select(x => x).OrderBy(x => rnd.Next()).Take(batch_size).ToList();
      return selected;
    }

    public int size
    {
      get
      {
        return this.memory.Count;
      }
    }
  }

  
}
