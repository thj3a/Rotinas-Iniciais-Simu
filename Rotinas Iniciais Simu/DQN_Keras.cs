using DequeNet;
using System;
using System.Collections.Generic;
using System.Collections;
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
using TorchSharp;

namespace RotinasIniciais
{

    public class dqn_keras
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
        private Deque<(NDArray, int, float, NDArray, bool)> memory;
        Random rnd = new Random();
        private Sequential model;


        public dqn_keras(int memory_size, int batch_size, double gamma, double exploration_max, double exploration_min, double exploration_decay, double learning_rate, double tau, int n_actions, int n_inputs)
        {
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
            this.model = create_model_keras(2, 24, n_inputs, n_actions);

        }
        public virtual int get_action(NDArray state, bool should_explore = true)
        {
            int action = -1;
            if (should_explore)
            {
                exploration_max *= exploration_decay;
                exploration_max = Math.Max(exploration_min, exploration_max);
                if (rnd.NextDouble() < exploration_max)
                    return rnd.Next(2);
            }

            var q_values = model.predict(state)[0].numpy()[0];
            var best_action = np.argmax(q_values);
            action = best_action[0];
            return action;
        }
        public virtual (NDArray, int, bool) step(NDArray state, int action)
        {
            NDArray next_state;
            var _state = state.ToArray<float>();
            var _next_state = state.ToArray<float>();
            var done = false;

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

            next_state = np.array(new float[,] { { _next_state[0], _next_state[1] } });
            int reward = calc_reward(state, next_state, done);
            return (next_state, reward, done);
        }
        public virtual int calc_reward(NDArray state, NDArray next_state, bool done)
        {
            var _state = state.ToArray<float>();
            var _next_state = next_state.ToArray<float>();

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
        public virtual (NDArray, bool) reset_env()
        {
            var done = false;
            NDArray state = np.array(new int[,] { { 2, 1 } }, TF_DataType.TF_FLOAT);

            return (state, done);

        }
        public void remember(NDArray state, int action, float reward, NDArray next_state, bool done)
        {
            this.memory.PushRight((state, action, reward, next_state, done));
        }
        public void replay()
        {
            if (this.memory.Count < this.batch_size) return;

            var samples = this.memory.OrderBy(x => rnd.Next()).Take(batch_size);

            foreach ((NDArray state, int action, float reward, NDArray next_state, bool done) in samples)
            {
                var target = this.model.predict(state)[0].numpy()[0];
                if (done) target[0][action] = reward;
                else
                {
                    var Q_future = np.amax(this.model.predict(next_state)[0].numpy()[0]);
                    target[0][action] = reward + Q_future * this.gamma;
                }
                model.fit(state, target, verbose: 0);
            }
        }
        static Sequential create_model_keras(int layers, int neurons, int input_size, int output_size)
        {
            // Prepare layers
            var list_layers = new List<ILayer>();
            list_layers.Add(keras.layers.Dense(neurons, keras.activations.Relu, input_shape: new Shape(input_size)));
            for (int i = 0; i < layers; i++) list_layers.Add(keras.layers.Dense(64, keras.activations.Relu));
            list_layers.Add(keras.layers.Dense(output_size, keras.activations.Linear));

            //Build sequential model
            var model = keras.Sequential(layers: list_layers);
            for (int i = 1, c = list_layers.Count; i < c; i++) model.Layers.Add(list_layers[i]);

            // Model Compile
            model.compile(optimizer: keras.optimizers.Adam(), loss: keras.losses.CategoricalCrossentropy(), metrics: new string[] { "accuracy" });
            model.summary();
            return model;
        }

        public void run(int episodes = 1000)
        {
            List<long> rewards = new();
            (NDArray state, bool done) = this.reset_env();
            NDArray next_state = state;
            int reward = 0;
            int action = -1;

            for (int ep = 0; ep < episodes; ep++)
            {
                (state, done) = this.reset_env();
                List<int> r = new();
                while (!done)
                {
                    action = this.get_action(state);
                    (next_state, reward, done) = this.step(state, action);
                    this.remember(state, action, reward, next_state, done);
                    if (ep % 32 == 0)
                        this.replay();
                    state = next_state;
                    r.Add(reward);
                }
                rewards.Add(r.Sum());
                // Console.WriteLine($"Next state: [{String.Join(",", state)}], Reward: {reward}, Done?: {done}");
                Console.WriteLine($"Episode {ep}, Reward: {rewards[ep]}");
            }
        }
    }
}

