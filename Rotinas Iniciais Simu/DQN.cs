using DequeNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Models;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace RotinasIniciais
{


  public class DQN
  {
    private int batch_size;
    private double gamma;
    private double exploration_max;
    private double exploration_min;
    private double exploration_decay;
    private double learning_rate;
    private double tau;
    private int n_actions;
    private int n_inputs;
    private List<float> loss_history;
    private int fit_count;
    private List<object> nodes_queue;
    private int memory_size;
    private Deque<(List<int>, int, float, List<int>, bool)> memory;
    Random rnd = new Random();
    private Sequential model;
    

    public DQN( int memory_size, int batch_size, double gamma,  double exploration_max, double exploration_min, double exploration_decay, double learning_rate, double tau, int n_actions, int n_inputs)
    {
      this.memory = new ();
      memory.Capacity = memory_size;
      this.batch_size = batch_size;
      this.gamma = gamma;
      this.exploration_max = exploration_max;
      this.exploration_min = exploration_min;
      this.exploration_decay = exploration_decay;
      this.learning_rate = learning_rate;
      this.tau = tau;
      this.loss_history = new ();
      this.fit_count = 0;
      this.nodes_queue = new ();
      this.model = CreateModel(2, 24, n_inputs, n_actions);

    }
    public virtual int get_action(List<int> state, bool should_explore = true)
    {
      int action = -1;
      if (should_explore)
      {
        exploration_max *= exploration_decay;
        exploration_max = Math.Max(exploration_min, exploration_max);
        if (rnd.NextDouble() < exploration_max) 
          return rnd.Next(2);
      }

      var q_values = model.predict(np.array(state.ToArray(), TF_DataType.TF_FLOAT).reshape(new Shape(1, 2)))[0].numpy()[0];
      var best_action = np.argmax(q_values);
      action = best_action[0];
      return action;
    }
    public virtual (List<int>, int, bool) step(List<int> state, int action)
    {
      List<int> next_state = new List<int>(state);

      if (action == 0)
      {
        next_state[0] -= 1;
        //Console.WriteLine("andei pra esquerda");
      }
      else
      {
        next_state[0] += 1;
        //Console.WriteLine("andei pra direita");
      }
      var done = false;
      if (next_state[0] == 0 || next_state[0] == 3)
      {
        done = true;
        //Console.WriteLine("caiu no fogo ou ganhou");

      }
      if (next_state[1] == 1 && next_state[0] == 1)
      {
        next_state[1] = 0;
        //Console.WriteLine("diamante parou de existir");
      }
      return (next_state, calc_reward(state, next_state, done), done);
    }
    public virtual int calc_reward(List<int> state, List<int> next_state, bool done)
    {
      if (next_state[0] == 0)
      {
        return -10;
      }
      else if (next_state[0] == 3)
      {
        return 10;
      }
      else if (state[1] == 1 && next_state[0] == 1)
      {
        return 3;
      }
      else
      {
        return 0;
      }
    }
    public virtual (List<int>, bool) reset_env()
    {
      var done = false;
      List<int> state = new List<int>();
      state.Add(2);
      state.Add(1);
      state[0] = 2;
      state[1] = 1;

      return (state, done);

    }
    public void remember(List<int> state, int action, float reward, List<int> next_state, bool done)
    {
      this.memory.PushRight((state, action, reward, next_state, done));
    }
    public void replay()
    {
      if (this.memory.Count < this.batch_size) return;

      var samples = this.memory.OrderBy(x => rnd.Next()).Take(batch_size);

      foreach ((List<int> state, int action, float reward, List<int> next_state, bool done) in samples)
      {
        var target = this.model.predict(np.array(state.ToArray(), TF_DataType.TF_FLOAT).reshape(new Shape(1, 2)))[0].numpy()[0];
        if (done) target[0][action] = reward;
        else
        {
          var Q_future = np.argmax(this.model.predict(np.array(next_state.ToArray(), TF_DataType.TF_FLOAT).reshape(new Shape(1, 2)))[0].numpy()[0]);
          target[0][action] = reward + Q_future * this.gamma;
        }
        var np_state = np.array(state.ToArray(), TF_DataType.TF_FLOAT).reshape(new Shape(1, 2));
        model.fit(np_state, target, verbose: 0);
      }

    }
    static Sequential CreateModel(int layers, int neurons, int input_size, int output_size)
    {
      // Prepare layers
      var list_layers = new List<ILayer>();
      list_layers.Add(keras.layers.Dense(neurons, keras.activations.Relu, input_shape: new Shape(input_size)));
      for (int i = 0; i < layers; i++) list_layers.Add(keras.layers.Dense(64, keras.activations.Relu));
      list_layers.Add(keras.layers.Dense(output_size, keras.activations.Softmax));

      //Build sequential model
      var model = keras.Sequential(layers: list_layers);
      for (int i = 1, c = list_layers.Count; i < c; i++) model.Layers.Add(list_layers[i]);

      // Model Compile
      model.compile(optimizer: keras.optimizers.Adam(), loss: keras.losses.CategoricalCrossentropy(), metrics: new string[] { "accuracy" });
      model.summary();
      return model;
    }
  }
}

