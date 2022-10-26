using DequeNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.TensorExtensionMethods;
using TorchSharp.Modules;
using System.ComponentModel.DataAnnotations;
using System.Runtime.CompilerServices;

namespace RotinasIniciais
{

  public class dqn_torch
  {
    private int batch_size;
    private double gamma;
    private decimal exploration_max;
    private decimal exploration_min;
    private decimal exploration_decay;
    private double learning_rate;
    private double tau;
    private int n_actions;
    private int n_inputs;
    private int layers;
    private int neurons;
    private List<float> loss_history;
    private int fit_count;
    private List<object> nodes_queue;
    private int memory_size;
    private List<(torch.Tensor, int, float, torch.Tensor, bool)> memory;
    Random rnd = new Random();
    private Sequential model;
    Adam optimizer;
    Linear lin1;
    Linear lin2;
    Linear lin3;
    Linear lin4;
    string device;
    
    public dqn_torch(int memory_size, int batch_size, double gamma, decimal exploration_max, decimal exploration_min, decimal exploration_decay, float learning_rate, double tau, int n_actions, int n_inputs, int layers, int neurons)
    {
      this.device = "cpu";
      this.memory = new();
      memory.Capacity = memory_size;
      this.batch_size = batch_size;
      this.gamma = gamma;
      this.exploration_max = exploration_max;
      this.exploration_min = exploration_min;
      this.exploration_decay = exploration_decay;
      this.learning_rate = learning_rate;
      this.tau = tau;
      this.loss_history = new();
      this.fit_count = 0;
      this.nodes_queue = new();
      this.n_inputs = n_inputs;
      this.n_actions = n_actions;
      this.neurons = neurons;
      this.layers = layers;
      this.lin1 = Linear(n_inputs, neurons);
      this.lin2 = Linear(neurons, neurons);
      this.lin3 = Linear(neurons, neurons);
      this.lin4 = Linear(neurons, n_actions);
      this.model = create_model_torch(layers, neurons, n_inputs, n_actions).to(device);
      this.optimizer = torch.optim.Adam(model.parameters(), learning_rate);
      model.eval();
      // this.optimizer = torch.optim.SGD(model.parameters(), learning_rate);

    }
    public virtual int get_action(torch.Tensor state, bool should_explore = true)
    {
      int action = -1;
      if (should_explore)
      {
        exploration_max *= exploration_decay;
        exploration_max = Math.Max(exploration_min, exploration_max);
        if (rnd.NextDouble() < (double)exploration_max)
          return rnd.Next(2);
      }

      var q_values = model.forward(state)[0];
      var best_action = torch.argmax(q_values);
      
      action = (int)best_action.item<long>();
      return action;
    }
    
    public virtual (torch.Tensor, int, bool) step(torch.Tensor state, int action)
    {
      torch.Tensor next_state;
      var _state = new List<float>();
      var _next_state = new List<float>();
      var done = false;

      for(long i=0, imax = state.size()[0]+1; i < imax; i++)
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
    public virtual int calc_reward(torch.Tensor state, torch.Tensor next_state, bool done)
    {
      var _state = new List<float>();
      var _next_state = new List<float>();
      for (long i=0, imax = state.size()[0]+1; i<imax; i++)
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
    public virtual (torch.Tensor, bool) reset_env()
    {
      var done = false;
      torch.Tensor state = torch.tensor(new float[,] { { 2, 1 } });

      return (state, done);

    }
    public void remember(torch.Tensor state, int action, float reward, torch.Tensor next_state, bool done)
    {
      this.memory.Add((state, action, reward, next_state, done));
    }
    public float replay()
    {
      if (this.memory.Count < this.batch_size) return -1f;
      var memory_size = memory.Count;
      var samples = new List<(torch.Tensor, int, float, torch.Tensor, bool)>();
      //for (int i=0; i < this.batch_size; i++)
      //{
      //  samples.Add(memory[rnd.Next(memory_size)]);
      //}
      
      samples = this.memory.Select(x=>x).OrderBy(x => rnd.Next()).Take(batch_size).ToList();
      var loss = new List<float>();

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

      var _states = new float[32, 2];
      var _targets = new float[32, 2];
      for (int i = 0, max = samples.Count(); i < max; i++)
      {
        _states[i, 0] = samples.ElementAt(i).Item1[0][0].item<float>();
        _states[i, 1] = samples.ElementAt(i).Item1[0][1].item<float>();
      }

      var states = torch.tensor(_states);
      var y_hat = model.forward(states);
      var y = model.forward(states);

      for (int i = 0, imax = samples.Count(); i < imax; i++)
      {
        if (samples.ElementAt(i).Item5)
        {
          y[i][samples.ElementAt(i).Item2] = samples.ElementAt(i).Item3;
        }
        else
        {
          var next_state_q_values = model.forward(samples.ElementAt(i).Item4);
          var next_state_q_value = torch.max(next_state_q_values, 1).values;
          y[i][samples.ElementAt(i).Item2] = samples.ElementAt(i).Item3 + (this.gamma * next_state_q_value.item<float>());
        }
      }

      var output = functional.mse_loss(y_hat, y);
      model.zero_grad();
      output.backward();
      optimizer.step();
      return output.item<float>();
    }
    Sequential create_model_torch(int layers, int neurons, int input_size, int output_size)
    {

      //Sequential model = Sequential();
      //model.append(lin1);
      //model.append(ReLU());
      //model.append(lin2);
      //model.append(ReLU());
      //model.append(lin3);

      Sequential model = Sequential((this.lin1), (ReLU()), (this.lin2), (ReLU()), (this.lin3), (ReLU()), (this.lin4));
      return model;
    }

    public void run(int episodes = 1000)
    {
      List<int> rewards = new();
      List<float> losses = new();
      (torch.Tensor state, bool done) = this.reset_env();
      torch.Tensor next_state = state;
      int reward = 0;
      int action = -1;

      for (int ep = 0; ep < episodes; ep++)
      {
        (state, done) = this.reset_env();
        List<int> r = new();
        List<float> l = new();
        while (!done)
        {
          action = this.get_action(state.clone());
          (next_state, reward, done) = this.step(state.clone(), action);
          this.remember(state.clone(), action, reward, next_state.clone(), done);
          //if (ep % 32 == 0)
          l.Add(this.replay());
          state = next_state.clone();
          r.Add(reward);
        }
        rewards.Add(r.Sum());
        losses.Add(torch.mean(torch.tensor(l)).item<float>());
        Console.Write("Rewards in ep.: ");
        r.ForEach(x=> Console.Write($"{x.ToString()}..."));
        Console.WriteLine();
        Console.Write("Losses in ep.:");
        l.ForEach(x => Console.Write($"{x.ToString()}..."));
        // Console.WriteLine($"Next state: [{String.Join(",", state)}], Reward: {reward}, Done?: {done}");
        Console.WriteLine();
        Console.WriteLine($" --- Episode {ep}, Ep. Reward: {rewards[ep]}, Ep. Loss: {losses[ep]}, Exploration: {exploration_max}");
      }
    }
  }
}

