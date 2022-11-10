//using Tensorflow;
//using Tensorflow.Keras;
//using Tensorflow.Keras.ArgsDefinition;
//using Tensorflow.Keras.Engine;
//using Tensorflow.Keras.Layers;
//using Tensorflow.Keras.Models;
//using Tensorflow.NumPy;
//using static Tensorflow.Binding;
//using static Tensorflow.KerasApi;
//using Keras.Models;

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
using Python.Runtime;

Random rnd = new();
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
int episodes = (int)1e3;
var device = torch.cuda.is_available() ? torch.device("CUDA") : torch.device("cpu");

//dqn_torch model = new dqn_torch(batch_size, gamma, exploration_max, exploration_min, exploration_decay, learning_rate, tau, n_outputs, n_inputs, layers, neurons, "model", device);

//dqn_torch aux_model = new dqn_torch(batch_size, gamma, exploration_max, exploration_min, exploration_decay, learning_rate, tau, n_outputs, n_inputs, layers, neurons, "aux_model", device);

var lin1 = Linear(n_inputs, neurons);
var lin2 = Linear(neurons, neurons);
var lin3 = Linear(neurons, n_outputs);

var seq = new dqn_torch();
var aux = new dqn_torch();
var loss_func = MSELoss();
Adam optimizer = new(seq.parameters(), learning_rate);

simple_env env = new simple_env();
replay_memory memory = new replay_memory(memory_size);

List<int> list_rewards = new();
List<float> list_losses = new();
(float[,] state, bool done) = env.reset();
var next_state = new float[,] { { state[0,0], state[0,1] } };
int reward = 0;
int action = -1;

for (int ep = 0; ep < episodes; ep++)
{
  (state, done) = env.reset();
  List<int> r = new();
  List<float> l = new();
  while (!done)
  {
    action = get_action(state, seq, rnd);
    (next_state, reward, done) = env.step(state, action);
    memory.remember(state, action, reward, next_state, done);

    if (!(memory.size < batch_size+100))
    {

      List<(float[,] state, int action, float reward, float[,] next_state, bool done)> samples = memory.sample(batch_size);

      List<Tensor> Qs_current = new();
      List<Tensor> Qs_expected = new();
      foreach (var sample in samples)
      {
        
        //var stateT = tensor(new float[,] { { sample.state[0][0].item<float>(), sample.state[0][1].item<float>() } }, dtype: ScalarType.Float32, device: torch.CPU);
        //var next_stateT = tensor(new float[,] { { sample.next_state[0][0].item<float>(), sample.next_state[0][1].item<float>() } }, dtype: ScalarType.Float32, device: torch.CPU);
        //var actionT = tensor(new float[] { sample.action }, dtype: ScalarType.Float32, device: torch.CPU);
        //var rewardT = tensor(new float[] { sample.reward }, dtype: ScalarType.Float32, device: torch.CPU);
        //var doneT = tensor(new float[] { sample.done ? 1.0f : 0.0f }, dtype: ScalarType.Float32, device: torch.CPU);

        var stateT = tensor(new float[,] { { sample.state[0,0], sample.state[0,1] } }, dtype: ScalarType.Float32, device: torch.CPU);
        var next_stateT = tensor(new float[,] { { sample.next_state[0,0], sample.next_state[0,1]} }, dtype: ScalarType.Float32, device: torch.CPU);
        var actionT = tensor(new float[] { sample.action }, dtype: ScalarType.Float32, device: torch.CPU);
        var rewardT = tensor(new float[] { sample.reward }, dtype: ScalarType.Float32, device: torch.CPU);
        var doneT = tensor(new float[] { sample.done ? 1.0f : 0.0f }, dtype: ScalarType.Float32, device: torch.CPU);
        var done_float = sample.done ? 1.0f : 0.0f;

        var Q_future = aux.forward(stateT).max().item<float>();
        var Q_Expected = aux.forward(stateT);
        Q_Expected[0][sample.action] = sample.reward + ((gamma * Q_future) * (1 - done_float));
        var Q_current = seq.forward(stateT);

        Qs_current.Add(Q_current);
        Qs_expected.Add(Q_Expected);
      }
      //stateT.print(); actionT.print(); rewardT.print(); doneT.print();
      //Q_Expected.print(); Q_current.print();

      var Q_currentT = torch.cat(Qs_current);
      var Q_expectedT = torch.cat(Qs_expected);

      var loss = functional.mse_loss(Q_currentT, Q_expectedT, Reduction.Sum);
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
      l.Add(loss.item<float>());
      

      //Tensor states = torch.cat(samples.Select(x => x.state).ToList());
      //Tensor next_states = torch.cat(samples.Select(x => x.next_state.clone()).ToList());
      //Tensor actions = torch.tensor(samples.Select(x => (long)x.action).ToList());
      //Tensor rewards = torch.tensor(samples.Select(x => x.reward).ToList());
      //Tensor dones = torch.tensor(samples.Select(x => x.done).ToList());

      //Tensor expected = torch.zeros(new long[] { batch_size, 2 });
      //Tensor pred = aux.forward(states);
      //for (int i = 0; i < batch_size; i++)
      //{

      //  if (dones[i].item<bool>())
      //  {
      //    expected[i] = pred[i];
      //    expected[i][actions[i]] = rewards[i];
      //  }
      //  else
      //  {
      //    expected[i] = pred[i];
      //    var q_future = aux.forward(next_states[i]).max().item<float>();
      //    expected[i][actions[i]] = rewards[i] + (gamma * q_future);
      //  }
      //}

      //using var loss = loss_func.forward(seq.forward(states), expected);
      //l.Add(loss.item<float>());

      //seq.zero_grad();
      //loss.backward();
      //optimizer.step();

      //pred.print();
      //expected.print();
      //loss.print();
    }

    state = new float[,] { { next_state[0,0], next_state[0,1] } };
    r.Add(reward);
  }
  list_rewards.Add(r.Sum());
  list_losses.Add(torch.mean(torch.tensor(l)).item<float>());

  Console.WriteLine($" --- Episode {ep} --- Sum of Ep. Rewards: {list_rewards[ep]}, Mean Ep. Loss: {list_losses[ep]}, Exploration: {exploration_max}");

  aux.load_state_dict(seq.state_dict());
}

float replay(replay_memory memory, dqn_torch model, Module<Tensor, Tensor> aux_model, optim.Optimizer optimizer)
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

  if (memory.size < batch_size) return -1f;
  
  var loss = new List<float>();
  var _states = new float[32, 2];
  var _targets = new float[32, 2];
  
  List<(float[,] state, int action, float reward, float[,] next_state, bool done)> samples = memory.sample(batch_size);
  var states = torch.cat(samples.Select(x => tensor(x.state)).ToList());

  //for (int i = 0, max = samples.Count(); i < max; i++)
  //{
  //  _states[i, 0] = samples[i].state[0][0].item<float>();
  //  _states[i, 1] = samples[i].state[0][1].item<float>();
  //}
  //var states = torch.tensor(_states);

  
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
      var next_state_q_values = aux_model.forward(tensor(samples[i].next_state));
      var next_state_q_value = torch.max(next_state_q_values, 1).values.item<float>();
      targets[i][samples[i].action] = samples[i].reward + (gamma * next_state_q_value);
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

int get_action(float[,] state, dqn_torch nnet, Random rnd, bool should_explore = true)
{
  int action = -1;
  Tensor q_values;
  if (should_explore)
  {
    exploration_max *= exploration_decay;
    exploration_max = Math.Max(exploration_min, exploration_max);
    if (torch.rand(1).item<float>() < exploration_max)
      return rnd.Next(2);
  }
  using (torch.no_grad())
  {
    q_values = nnet.forward(tensor(state))[0];
  }
  var best_action = torch.argmax(q_values);

  action = (int)best_action.item<long>();
  return action;
}
class dqn_torch : Module<Tensor, Tensor>
{
  //public int batch_size;
  //public float gamma;
  //public float exploration_max;
  //public float exploration_min;
  //public float exploration_decay;
  //public float learning_rate;
  //public float tau;
  //public int n_actions;
  //public int n_inputs;
  //public int layers;
  //public int neurons;

  public Random rnd = new Random();
  public Module<Tensor, Tensor> lin1 = Linear(2, 24);
  public Module<Tensor, Tensor> lin2 = Linear(24, 24);
  public Module<Tensor, Tensor> lin3 = Linear(24, 2);

  public dqn_torch(
                   //int batch_size,
                   //float gamma,
                   //float exploration_max,
                   //float exploration_min,
                   //float exploration_decay,
                   //float learning_rate,
                   //float tau,
                   //int n_outputs,
                   //int n_inputs,
                   //int layers,
                   //int neurons,
                   //string name,
                   torch.Device device = null) : base(nameof(dqn_torch))
  {
    RegisterComponents();

    if (device != null && device.type == DeviceType.CUDA)
      this.to(device);

    //this.batch_size = batch_size;
    //this.gamma = gamma;
    //this.exploration_max = exploration_max;
    //this.exploration_min = exploration_min;
    //this.exploration_decay = exploration_decay;
    //this.learning_rate = learning_rate;
    //this.tau = tau;
    //this.n_inputs = n_inputs;
    //this.n_actions = n_outputs;
    //this.neurons = neurons;
    //this.layers = layers;
  }

  //public int get_action(Tensor state, bool should_explore = true)
  //{
  //  int action = -1;
  //  Tensor q_values;
  //  if (should_explore)
  //  {
  //    exploration_max *= exploration_decay;
  //    exploration_max = Math.Max(exploration_min, exploration_max);
  //    if (this.rnd.NextDouble() < exploration_max)
  //      return rnd.Next(2);
  //  }
  //  using (torch.no_grad())
  //  {
  //    q_values = this.forward(state)[0];
  //  }
  //  var best_action = torch.argmax(q_values);

  //  action = (int)best_action.item<long>();
  //  return action;
  //}

  public override Tensor forward(Tensor input)
  {
    using var x1 = lin1.forward(input);
    using var x2 = relu (x1);
    using var x3 = lin2.forward(x2);
    using var x4 = relu(x3);
    return lin3.forward(x4);
  }
}

public class simple_env
{
  public int calc_reward(float[,] state, float[,] next_state, bool done)
  {
    //var _state = new List<float>();
    //var _next_state = new List<float>();
    //for (long i = 0, imax = state.size()[0] + 1; i < imax; i++)
    //{
    //  _state.Add(state[0][i].item<float>());
    //}
    //for (long i = 0, imax = next_state.size()[0]; i < imax; i++)
    //{
    //  _next_state.Add(next_state[0][i].item<float>());
    //}

    if (next_state[0,0] == 0)
    {
      return -10;
    }
    else if (next_state[0,0] == 3)
    {
      return 10;
    }
    else if (state[0,1] == 1 && next_state[0,0] == 1)
    {
      return 3;
    }
    else
    {
      return 0;
    }
  }

  public (float[,], bool) reset()
  {
    var done = false;
    float[,] state = new float[,] { { 2, 1 } };

    return (state, done);

  }

  public (float[,], int, bool) step(float[,] state, int action)
  {
    float[,] next_state = new float[,] { { state[0,0], state[0,1] } };
    //var _state = new List<float>();
    //var _next_state = new List<float>();
    var done = false;

    //for (long i = 0, imax = state.size()[0] + 1; i < imax; i++)
    //{
    //  _state.Add(state[0][i].item<float>());
    //  _next_state.Add((float)state[0][i].item<float>());
    //}

    if (action == 0)
    {
      next_state[0,0] -= 1;
      //Console.WriteLine("andei pra esquerda");
    }
    else
    {
      next_state[0,0] += 1;
      //Console.WriteLine("andei pra direita");
    }

    if (next_state[0,0] == 0 || next_state[0,0] == 3)
    {
      done = true;
      //Console.WriteLine("caiu no fogo ou ganhou");
    }
    if (next_state[0,1] == 1 && next_state[0,0] == 1)
    {
      next_state[0,1] = 0;
      //Console.WriteLine("diamante parou de existir");
    }
    //next_state = torch.tensor(new float[,] { { _next_state[0], _next_state[1] } });
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
  public List<(float[,] state, int action, float reward, float[,] next_state, bool done)> memory;
  public Random rnd;
  public replay_memory(int capacity = 2000)
  {
    // For now, capacity is not working... Improve this later.
    memory = new();
    rnd = new();
  }
  public void remember(float[,] state, int action, float reward, float[,] next_state, bool done)
  {
    this.memory.Add((state, action, reward, next_state, done));
  }

  public List<(float[,], int, float, float[,], bool)> sample(int batch_size)
  {
    List<(float[,] state, int action, float reward, float[,] next_state, bool done)> selected = this.memory.Select(x => x).OrderBy(x => rnd.Next()).Take(batch_size).ToList();
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


