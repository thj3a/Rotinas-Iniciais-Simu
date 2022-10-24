//using DequeNet;
//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;
//using TorchSharp;
//using static TorchSharp.torch.nn;
//using static TorchSharp.TensorExtensionMethods;
//using TorchSharp.Modules;

//namespace RotinasIniciais
//{

//  public class DQN_torch
//  {
//    private int batch_size;
//    private double gamma;
//    private double exploration_max;
//    private double exploration_min;
//    private double exploration_decay;
//    private double learning_rate;
//    private double tau;
//    private int n_actions;
//    private int n_inputs;
//    private List<float> loss_history;
//    private int fit_count;
//    private List<object> nodes_queue;
//    private int memory_size;
//    private Deque<(torch.Tensor, int, float, torch.Tensor, bool)> memory;
//    Random rnd = new Random();
//    private Sequential model;
    

//    public DQN_torch( int memory_size, int batch_size, double gamma,  double exploration_max, double exploration_min, double exploration_decay, double learning_rate, double tau, int n_actions, int n_inputs)
//    {
//      this.memory = new ();
//      memory.Capacity = memory_size;
//      this.batch_size = batch_size;
//      this.gamma = gamma;
//      this.exploration_max = exploration_max;
//      this.exploration_min = exploration_min;
//      this.exploration_decay = exploration_decay;
//      this.learning_rate = learning_rate;
//      this.tau = tau;
//      this.loss_history = new ();
//      this.fit_count = 0;
//      this.nodes_queue = new ();
//      this.model = CreateModel_Keras(2, 24, n_inputs, n_actions);

//    }
//    public virtual int get_action(torch.Tensor state, bool should_explore = true)
//    {
//      int action = -1;
//      if (should_explore)
//      {
//        exploration_max *= exploration_decay;
//        exploration_max = Math.Max(exploration_min, exploration_max);
//        if (rnd.NextDouble() < exploration_max) 
//          return rnd.Next(2);
//      }

//      var q_values = model.predict(state)[0].numpy()[0];
//      var best_action = np.argmax(q_values);
//      action = best_action[0];
//      return action;
//    }
//    public virtual (torch.Tensor, int, bool) step(torch.Tensor state, int action)
//    {
//      torch.Tensor next_state;
//      var _state = new List<float>();
//      var _next_state = new List<float>();
//      var done = false;

      
//      foreach (var i in state.ToMultiDimArray<float>())
//      {
//        _state.Add((float)i);
//        _next_state.Add((float)i);
//      }

//      if (action == 0)
//      {
//        _next_state[0] -= 1;
//        //Console.WriteLine("andei pra esquerda");
//      }
//      else
//      {
//        _next_state[0] += 1;
//        //Console.WriteLine("andei pra direita");
//      }
      
//      if (_next_state[0] == 0 || _next_state[0] == 3)
//      {
//        done = true;
//        //Console.WriteLine("caiu no fogo ou ganhou");
//      }
//      if (_next_state[1] == 1 && _next_state[0] == 1)
//      {
//        _next_state[1] = 0;
//        //Console.WriteLine("diamante parou de existir");
//      }

      
//      next_state = np.array(new float[,] { { _next_state[0], _next_state[1] } });

//      return (next_state, calc_reward(state, next_state, done), done);
//    }
//    public virtual int calc_reward(torch.Tensor state, torch.Tensor next_state, bool done)
//    {
//      var _state = new List<float>();
//      var _next_state = new List<float>();
//      foreach (var i in state.ToMultiDimArray<float>())
//      {
//        _state.Add((float)i);
//      }
//      foreach (var i in next_state.ToMultiDimArray<float>())
//      {
//        _next_state.Add((float)i);
//      }

//      if (_next_state[0] == 0)
//      {
//        return -10;
//      }
//      else if (_next_state[0] == 3)
//      {
//        return 10;
//      }
//      else if (_state[1] == 1 && _next_state[0] == 1)
//      {
//        return 3;
//      }
//      else
//      {
//        return 0;
//      }
//    }
//    public virtual (torch.Tensor, bool) reset_env()
//    {
//      var done = false;
//      NDArray state = np.array(new int[,] { { 2, 1 } }, TF_DataType.TF_FLOAT);

//      return (state, done);

//    }
//    public void remember(NDArray state, int action, float reward, NDArray next_state, bool done)
//    {
//      this.memory.PushRight((state, action, reward, next_state, done));
//    }
//    public void replay()
//    {
//      if (this.memory.Count < this.batch_size) return;

//      var samples = this.memory.OrderBy(x => rnd.Next()).Take(batch_size);
      
//      foreach ((NDArray state, int action, float reward, NDArray next_state, bool done) in samples)
//      {
//        var target = this.model.predict(state)[0].numpy()[0];
//        if (done) target[0][action] = reward;
//        else
//        {
//          var Q_future = np.argmax(this.model.predict(next_state)[0].numpy()[0]);
//          target[0][action] = reward + Q_future * this.gamma;
//        }
//        model.fit(state, target, verbose: 0);
//      }

//    }
//    static Sequential CreateModel_torch(int layers, int neurons, int input_size, int output_size)
//    {

//      var model = Sequential();
//      model.append(Linear(input_size, neurons));
//      for (int i = 0; i < layers; i++)
//      {
//        model.append(ReLU());
//      }
//      model.append(Linear(neurons, output_size));
//      return model;
//    }

//  }
//}

